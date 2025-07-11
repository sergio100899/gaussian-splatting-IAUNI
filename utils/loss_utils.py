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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from PIL import Image
from transformers import pipeline
import torchvision.transforms as transforms

pipe = pipeline(
    task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf"
)

try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01**2
C2 = 0.03**2


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def sobel_edges(img: torch.tensor) -> torch.tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)  # Asegura batch dimension

    if img.shape[1] != 1:
        # Forzamos escala de grises (por si es RGB)
        img = img[:, 0:1] * 0.2989 + img[:, 1:2] * 0.5870 + img[:, 2:3] * 0.1140

    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=img.device
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32, device=img.device
    ).unsqueeze(0)

    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    edges = torch.sqrt(grad_x**2 + grad_y**2)

    edges_norm = edges / (edges.max() + 1e-8)
    return edges_norm


def depth_inference(img: torch.tensor) -> torch.tensor:
    img = (img * 255).byte()
    img = img.permute(1, 2, 0)
    imagen_pil = Image.fromarray(img.cpu().numpy())

    depth = pipe(imagen_pil)["depth"]
    depth_tensor = transforms.ToTensor()(depth)

    return depth_tensor


def normal_inference(img: torch.tensor) -> torch.tensor:
    # depth_tensor = depth_inference(img).squeeze(0).to("cuda")

    # Calcular el mapa de normales
    zy, zx = torch.gradient(img.float(), dim=[0, 1])

    normal_map = torch.stack([-zx, -zy, torch.ones_like(img)], dim=2)

    # Normalización
    norm = torch.linalg.norm(normal_map, dim=2, keepdim=True) + 1e-8
    normal_map = normal_map / norm

    return normal_map


def l1_edge_loss(
    network_output: torch.tensor, gt: torch.tensor, alpha_edge: float = 0.2
):
    edges = sobel_edges(gt)
    edge_weight = 1.0 + alpha_edge * edges
    loss = edge_weight * torch.abs(network_output - gt)
    return loss.mean()


def edge_loss(network_output: torch.tensor, gt: torch.tensor):
    gt_edge = sobel_edges(gt)
    nt_edge = sobel_edges(network_output)
    return torch.abs((nt_edge - gt_edge)).mean()


def depth_loss(network_output: torch.tensor, gt: torch.tensor):
    gt_depth = depth_inference(gt)
    nt_depth = depth_inference(network_output)
    return torch.abs((nt_depth - gt_depth)).mean()


def edge_aware_normal_loss(I, D):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device)/4

    dD_dx = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_x, padding=1) for i in range(D.shape[0])])
    dD_dy = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_y, padding=1) for i in range(D.shape[0])])
    
    dI_dx = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_x, padding=1) for i in range(I.shape[0])])
    dI_dx = torch.mean(torch.abs(dI_dx), 0, keepdim=True)
    dI_dy = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_y, padding=1) for i in range(I.shape[0])])
    dI_dy = torch.mean(torch.abs(dI_dy), 0, keepdim=True)

    weights_x = (dI_dx-1)**500
    weights_y = (dI_dy-1)**500

    loss_x = abs(dD_dx) * weights_x
    loss_y = abs(dD_dy) * weights_y
    loss = (loss_x + loss_y).norm(dim=0, keepdim=True)
    return loss.mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
