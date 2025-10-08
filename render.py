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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.loss_utils import (
    depth_inference,
    align_mean_std_and_compute_loss,
    sobel_edges,
    depth_normal_consistency_loss,
)
from utils.image_utils import depth_to_normal
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
):
    # Variables para acumular losses en test
    total_depth_loss = 0.0
    total_normal_loss = 0.0
    total_normal_cons_loss = 0.0
    total_edge_loss = 0.0
    num_views = 0
    individual_losses = []  # Lista para guardar losses individuales
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "render_depth"
    )
    render_normals_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "render_normal"
    )
    render_edges_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "render_edges"
    )
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gts_depth_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_depth"
    )
    gt_normals_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_normal"
    )
    gt_edges_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_edges"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normals_path, exist_ok=True)
    makedirs(render_edges_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gts_depth_path, exist_ok=True)
    makedirs(gt_normals_path, exist_ok=True)
    makedirs(gt_edges_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
        )
        rendering = rendering_pkg["render"]
        rendering_depth = rendering_pkg[
            "depth"
        ]  # Usar la profundidad del render principal

        # Obtener profundidad y normales renderizadas directamente, como en train.py
        _, rendering_normal_map = gaussians.render_depth_and_normal(view)

        # Calcular edges para comparación
        rendering_edges = sobel_edges(rendering).squeeze()
        gt = view.original_image[0:3, :, :]
        gt_depth = depth_inference(gt).squeeze(0).to("cuda")
        gt_normals = depth_to_normal(view, gt_depth).squeeze()
        gt_edges = sobel_edges(gt).squeeze()

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2 :]
            gt = gt[..., gt.shape[-1] // 2 :]

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            rendering_depth,
            os.path.join(render_depth_path, "{0:05d}".format(idx) + ".png"),
        )
        torchvision.utils.save_image(
            gt_depth, os.path.join(gts_depth_path, "{0:05d}".format(idx) + ".png")
        )
        # Guardar el mapa de normales renderizado directamente
        rendering_normal_save = (rendering_normal_map + 1.0) / 2.0
        torchvision.utils.save_image(
            rendering_normal_save,
            os.path.join(render_normals_path, "{0:05d}".format(idx) + ".png"),
        )

        gt_normals = gt_normals.permute(2, 0, 1)
        gt_normals = (gt_normals + 1.0) / 2.0
        gt_normals = torch.clamp(gt_normals, 0.0, 1.0)
        torchvision.utils.save_image(
            gt_normals,
            os.path.join(gt_normals_path, "{0:05d}".format(idx) + ".png"),
        )

        # Guardar edges (ya están normalizados por sobel_edges)
        torchvision.utils.save_image(
            rendering_edges,
            os.path.join(render_edges_path, "{0:05d}".format(idx) + ".png"),
        )
        torchvision.utils.save_image(
            gt_edges,
            os.path.join(gt_edges_path, "{0:05d}".format(idx) + ".png"),
        )

        # Calcular losses solo para el conjunto de test
        if name == "test":
            # Calcular depth loss usando la misma lógica que train.py
            depth_loss = align_mean_std_and_compute_loss(
                rendering_depth.squeeze(), gt_depth
            )

            # Calcular depth-normal consistency loss, como en train.py
            normal_cons_loss = depth_normal_consistency_loss(
                rendering_depth, rendering_normal_map, view
            )

            # Calcular edge loss (usando los edges ya calculados)
            edge_loss_value = torch.abs((rendering_edges - gt_edges)).mean()

            # La 'normal_loss' original ahora es la 'normal_cons_loss'
            normal_loss = normal_cons_loss

            # Acumular losses
            total_depth_loss += depth_loss.item()
            total_normal_cons_loss += normal_cons_loss.item()
            total_edge_loss += edge_loss_value.item()
            num_views += 1

            # Guardar loss individual para esta imagen
            individual_losses.append(
                {
                    "image_idx": idx,
                    "depth_loss": depth_loss.item(),
                    "normal_cons_loss": normal_cons_loss.item(),
                    "edge_loss": edge_loss_value.item(),
                }
            )

    # Guardar losses en archivo txt solo para test
    if name == "test" and num_views > 0:
        avg_depth_loss = total_depth_loss / num_views
        avg_normal_cons_loss = total_normal_cons_loss / num_views
        avg_edge_loss = total_edge_loss / num_views

        # Crear directorio para losses si no existe
        losses_dir = os.path.join(model_path, name, "ours_{}".format(iteration))
        makedirs(losses_dir, exist_ok=True)

        # Guardar losses en archivo txt
        losses_file = os.path.join(losses_dir, "test_losses.txt")
        with open(losses_file, "w") as f:
            f.write(f"Test Losses for iteration {iteration}\n")
            f.write(f"Number of views: {num_views}\n")
            f.write("=" * 50 + "\n")
            f.write("INDIVIDUAL IMAGE LOSSES:\n")
            f.write("=" * 50 + "\n")

            # Escribir losses individuales para cada imagen
            for loss_info in individual_losses:
                f.write(f"Image {loss_info['image_idx']:05d}:\n")
                f.write(f"  Depth Loss:  {loss_info['depth_loss']:.6f}\n")
                f.write(f"  Normal Cons Loss: {loss_info['normal_cons_loss']:.6f}\n")
                f.write(f"  Edge Loss:   {loss_info['edge_loss']:.6f}\n")
                f.write("-" * 30 + "\n")

            f.write("\nSUMMARY:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Average Depth Loss:  {avg_depth_loss:.6f}\n")
            f.write(f"Average Normal Cons Loss: {avg_normal_cons_loss:.6f}\n")
            f.write(f"Average Edge Loss:   {avg_edge_loss:.6f}\n")
            f.write(f"Total Depth Loss:    {total_depth_loss:.6f}\n")
            f.write(f"Total Normal Cons Loss:   {total_normal_cons_loss:.6f}\n")
            f.write(f"Total Edge Loss:     {total_edge_loss:.6f}\n")

        print(f"Test losses saved to: {losses_file}")
        print(f"Average Depth Loss: {avg_depth_loss:.6f}")
        print(f"Average Normal Consistency Loss: {avg_normal_cons_loss:.6f}")
        print(f"Average Edge Loss: {avg_edge_loss:.6f}")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
    )
