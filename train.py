#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import (
    l1_loss,
    ssim,
    edge_loss,
    sobel_edges,
    depth_inference,
    edge_aware_normal_loss,
)  # noqa
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel]."
        )

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    depth_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(
        range(first_iter, opt.iterations), desc="Training progress"
    )  # progress bar for training iterations
    first_iter += 1

    L_depth = 0
    L_normal = torch.tensor(0.0, device="cuda")

    #Iteration start
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        loss = 0
        Ll1depth = 0

        gaussian_edge_indices = torch.tensor([], dtype=torch.long).to("cuda")

        for _ in range(2):

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            # print(viewpoint_cam.image_name) #image file name


            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            image, depth_map, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            # image =rendered_image
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)
            L_edge = edge_loss(image, gt_image)

            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            L_depth = 0
            L_normal = 0
            if iteration < opt.densify_until_iter:
                if iteration % 10 == 0:
                    # Rendereed Depth Map
                    depth_tensor_gpu = depth_map.squeeze().detach().to("cuda")
                    depth_np_norm = (depth_tensor_gpu - depth_tensor_gpu.min()) / (
                        depth_tensor_gpu.max() - depth_tensor_gpu.min()
                    )
                    depth_np_tensor = depth_np_norm.to(torch.float32)
                    # Ground Truth Depth Map
                    depth_tensor = depth_inference(gt_image).squeeze(0).to("cuda")
                    # Depth Loss
                    L_depth = l1_loss(depth_np_tensor, depth_tensor)

                    if iteration % 20 ==0:
                        normal_im_gt = depth_to_normal(depth_tensor, viewpoint_cam).squeeze()
                        normal_im_render = depth_to_normal(depth_map, viewpoint_cam).squeeze()

                        L_normal = l1_loss(normal_im_gt, normal_im_render)

                        normal_im_gt = normal_im_gt.cpu().numpy()  # Convertir a NumPy
                        normal_im_gt = (normal_im_gt - normal_im_gt.min()) / (normal_im_gt.max() - normal_im_gt.min())  # Normalizar
                        normal_im_gt = (normal_im_gt * 255).astype(np.uint8)  # Convertir a uint8
    
                        normal_im_render = normal_im_render.detach().cpu().numpy() # Convertir a NumPy
                        normal_im_render = (normal_im_render - normal_im_render.min()) / (normal_im_render.max() - normal_im_render.min())  # Normalizar
                        normal_im_render = (normal_im_render * 255).astype(np.uint8)  # Convertir a uint8
    
    
                        # image = Image.fromarray(normal_im_gt)
                        # image.save(f"./normals/gt/normals_gt_{iteration}.png")
    
                        # image = Image.fromarray(normal_im_render)
                        # image.save(f"./normals/rd/normals_rd_{iteration}.png")

                else:
                    L_depth = L_depth
                    L_normal = L_normal
            else:
                if iteration % 100 == 0:
                    # Rendereed Depth Map
                    depth_tensor_gpu = depth_map.squeeze().detach().to("cuda")

                    depth_np_norm = (depth_tensor_gpu - depth_tensor_gpu.min()) / (
                        depth_tensor_gpu.max() - depth_tensor_gpu.min()
                    )
                    depth_np_tensor = depth_np_norm.to(torch.float32)

                    # Ground Truth Depth Map
                    depth_tensor = depth_inference(gt_image).squeeze(0).to("cuda")
                    # Depth Loss
                    L_depth = l1_loss(depth_np_tensor, depth_tensor)

                    if iteration % 200 ==0:
                        normal_im_gt = depth_to_normal(depth_tensor, viewpoint_cam).squeeze()
                        normal_im_render = depth_to_normal(depth_map, viewpoint_cam).squeeze()

                        L_normal = l1_loss(normal_im_gt, normal_im_render)

                        normal_im_gt = normal_im_gt.cpu().numpy()  # Convertir a NumPy
                        normal_im_gt = (normal_im_gt - normal_im_gt.min()) / (normal_im_gt.max() - normal_im_gt.min())  # Normalizar
                        normal_im_gt = (normal_im_gt * 255).astype(np.uint8)  # Convertir a uint8
    
                        normal_im_render = normal_im_render.detach().cpu().numpy() # Convertir a NumPy
                        normal_im_render = (normal_im_render - normal_im_render.min()) / (normal_im_render.max() - normal_im_render.min())  # Normalizar
                        normal_im_render = (normal_im_render * 255).astype(np.uint8)  # Convertir a uint8
    
    
                        # image = Image.fromarray(normal_im_gt)
                        # image.save(f"./normals/gt/normals_gt_{iteration}.png")
    
                        # image = Image.fromarray(normal_im_render)
                        # image.save(f"./normals/rd/normals_rd_{iteration}.png")

                else:
                    L_depth = L_depth
                    L_normal = L_normal

            loss_cam = (
                (1.0 - opt.lambda_dssim) * Ll1
                + opt.lambda_dssim * (1.0 - ssim_value)
                + 0.2 * L_depth
                + 0.2 * L_edge
                + 0.2 * L_normal
            )

            loss += loss_cam
            Ll1depth += L_depth

            if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter:
                # ------------------------------------------
                # Edge indices
                # ------------------------------------------
                edges = sobel_edges(gt_image).squeeze()
                x = viewspace_point_tensor[:, 0].long()
                y = viewspace_point_tensor[:, 1].long()
                H, W = edges.shape
                valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                x_valid = x[valid]
                y_valid = y[valid]
                indices_valid = valid.nonzero(as_tuple=False).squeeze()
                edge_values = edges[y_valid, x_valid]
                edge_mask = edge_values > 0.7
                new_edge_indices = indices_valid[edge_mask].to("cuda")
                gaussian_edge_indices = torch.cat(
                    (gaussian_edge_indices, new_edge_indices), dim=0
                )
                gaussian_edge_indices = torch.unique(gaussian_edge_indices)


        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            # ema_Ll1depth_for_log = 0

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            training_report(
                tb_writer,
                iteration,
                # Ll1_edge,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (
                    pipe,
                    background,
                    1.0,
                    SPARSE_ADAM_AVAILABLE,
                    None,
                    dataset.train_test_exp,
                ),
                dataset.train_test_exp,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):

                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,  # min_opacity
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        gaussian_edge_indices=gaussian_edge_indices,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
