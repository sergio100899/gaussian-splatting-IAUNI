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

def align_mean_std_and_compute_loss(rendered_depth, gt_depth):
    """
    Alinea el mapa de profundidad renderizado con el ground truth usando la media y la desviación estándar,
    y luego calcula la pérdida L1. Esto evita el problema de la escala relativa.
    """
    gt_depth_detached = gt_depth.detach()

    mean_gt = gt_depth_detached.mean()
    std_gt = gt_depth_detached.std()

    mean_rendered = rendered_depth.mean()
    std_rendered = rendered_depth.std()


    aligned_depth = (rendered_depth - mean_rendered) / (std_rendered + 1e-6) * std_gt + mean_gt


    return l1_loss(aligned_depth, gt_depth_detached)

def align_mean_std_and_compute_gradient_loss(
    rendered_depth, gt_depth, gt_image, alpha=1.0
):
    """
    Alinea la profundidad renderizada con la del ground truth (media y std) y calcula una pérdida
    logarítmica ponderada por los bordes de la imagen ground truth. Esto ayuda a que la pérdida
    sea menos sensible en zonas con muchos detalles (bordes) y más en zonas planas.
    """
    gt_depth_detached = gt_depth.detach()

    mean_gt = gt_depth_detached.mean()
    std_gt = gt_depth_detached.std()

    mean_rendered = rendered_depth.mean()
    std_rendered = rendered_depth.std()

    aligned_depth = (rendered_depth - mean_rendered) / (
        std_rendered + 1e-6
    ) * std_gt + mean_gt

    gt_edges = sobel_edges(gt_image)

    depth_diff = torch.abs(gt_depth_detached - aligned_depth)

    gradient_loss = torch.exp(-gt_edges * alpha) * torch.log(1 + depth_diff)

    return gradient_loss.mean()

def cosine_similarity_loss(rendered_normals, gt_normals):
    """
    Calcula la pérdida como 1 menos la similitud de coseno entre los vectores normales.
    """
    if rendered_normals.shape[-1] != 3:
        # Si la forma es [C, H, W], permutar a [H, W, C]
        rendered_normals = rendered_normals.permute(1, 2, 0)
        gt_normals = gt_normals.permute(1, 2, 0)

    rendered_normals = torch.nn.functional.normalize(rendered_normals, p=2, dim=-1)
    gt_normals = torch.nn.functional.normalize(gt_normals, p=2, dim=-1)

    cosine_similarity = torch.sum(rendered_normals * gt_normals, dim=-1)

    loss = 1.0 - cosine_similarity

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

def eikonal_loss(grad_sdf):
    """
    L_eik = (||∇f(p)|| - 1)²
    """
    return (torch.linalg.norm(grad_sdf, dim=-1) - 1).pow(2).mean()

def normal_consistency_loss(normals, points, k=5):
    """
    L_cons = 1 - |n_i ⋅ n_j|
    """
    if points.shape[0] < k + 1:
        return torch.tensor(0.0, device=points.device)
    
    with torch.no_grad():
        dists = torch.cdist(points, points)
        knn_idx = torch.topk(dists, k + 1, largest=False, sorted=True).indices[:, 1:]

    neighbor_normals = normals[knn_idx.view(-1)].view(points.shape[0], k, 3)
    central_normals = normals.unsqueeze(1).expand(-1, k, -1)

    dot_product = torch.sum(central_normals * neighbor_normals, dim=2)
    
    return (1.0 - torch.abs(dot_product)).mean()

def curvature_loss(normals, points, k=5):
    """
    L_curv = ||n_i - n_j||²
    """
    if points.shape[0] < k + 1:
        return torch.tensor(0.0, device=points.device)

    with torch.no_grad():
        dists = torch.cdist(points, points)
        knn_idx = torch.topk(dists, k + 1, largest=False, sorted=True).indices[:, 1:]

    neighbor_normals = normals[knn_idx.view(-1)].view(points.shape[0], k, 3)
    central_normals = normals.unsqueeze(1).expand(-1, k, -1)

    return (central_normals - neighbor_normals).pow(2).sum(dim=-1).mean()
