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
from PIL import Image
import numpy as np


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_image_as_tensor_pil(path: str, grayscale: bool = False) -> torch.Tensor:
    """
    Carga una imagen con PIL y la convierte a torch.Tensor [C, H, W] normalizado a [0, 1].

    Args:
        path (str): Ruta a la imagen.
        grayscale (bool): Si True, la imagen se carga como 1 canal. Si False, como RGB (3 canales).

    Returns:
        torch.Tensor: Imagen normalizada en [C, H, W].
    """
    img = Image.open(path).convert("L" if grayscale else "RGB")
    img_np = np.array(img).astype(np.float32) / 255.0

    if grayscale:
        tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]
    else:
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]

    return tensor


# def depth_to_normal(depth_map: torch.tensor, camera: torch.tensor):
#     depth_map = depth_map.squeeze()
#     height, width = depth_map.shape
#     points_world = torch.zeros((height + 1, width + 1, 3)).to(depth_map.device)
#     points_world[:height, :width, :] = unproject_depth_map(depth_map, camera)

#     p1 = points_world[:-1, :-1, :]
#     p2 = points_world[1:, :-1, :]
#     p3 = points_world[:-1, 1:, :]

#     v1 = p2 - p1
#     v2 = p3 - p1

#     normals = torch.cross(v1, v2, dim=-1)
#     normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

#     return normals


def unproject_depth_map(depth_map: torch.tensor, camera: torch.tensor):
    depth_map = depth_map.squeeze()
    height, width = depth_map.shape
    x = torch.linspace(0, width - 1, width).cuda()
    y = torch.linspace(0, height - 1, height).cuda()
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Reshape the depth map and grid to N x 1
    depth_flat = depth_map.reshape(-1)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)

    # Normalize pixel coordinates to [-1, 1]
    X_norm = (X_flat / (width - 1)) * 2 - 1
    Y_norm = (Y_flat / (height - 1)) * 2 - 1

    # Create homogeneous coordinates in the camera space
    points_camera = torch.stack([X_norm, Y_norm, depth_flat], dim=-1)

    K_matrix = camera.projection_matrix

    # parse out f1, f2 from K_matrix
    f1 = K_matrix[2, 2]
    f2 = K_matrix[3, 2]

    # get the scaled depth
    sdepth = (f1 * points_camera[..., 2:3] + f2) / (points_camera[..., 2:3] + 1e-8)

    # concatenate xy + scaled depth
    points_camera = torch.cat((points_camera[..., 0:2], sdepth), dim=-1)
    points_camera = points_camera.view((height, width, 3))
    points_camera = torch.cat(
        [points_camera, torch.ones_like(points_camera[:, :, :1])], dim=-1
    )
    points_world = torch.matmul(points_camera, camera.full_proj_transform.inverse())

    # Discard the homogeneous coordinate
    points_world = points_world[:, :, :3] / points_world[:, :, 3:]
    points_world = points_world.view((height, width, 3))

    return points_world

def depths_to_points(view, depthmap):
    """Convierte un mapa de profundidad a una nube de puntos 3D."""
    c2w = view.world_view_transform.inverse()
    W, H = int(view.image_width), int(view.image_height)
    
    # Matriz de intrínsecos desde la matriz de proyección completa
    projection_matrix = view.full_proj_transform
    intrins = torch.zeros(3, 3, device="cuda")
    intrins[0, 0] = projection_matrix[0, 0] * W / 2
    intrins[1, 1] = projection_matrix[1, 1] * H / 2
    intrins[0, 2] = (projection_matrix[0, 2] + 1) * W / 2
    intrins[1, 2] = (projection_matrix[1, 2] + 1) * H / 2
    intrins[2, 2] = 1.0

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points_2d = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    
    # Desproyectar puntos
    rays_d_cam = points_2d @ intrins.inverse().T
    rays_d_world = rays_d_cam @ c2w[:3, :3].T
    rays_o_world = c2w[:3, 3]
    
    points_3d = depthmap.reshape(-1, 1) * rays_d_world + rays_o_world
    return points_3d.reshape(H, W, 3)

def depth_to_normal(view, depth):
    """Calcula el mapa de normales a partir de un mapa de profundidad."""
    points = depths_to_points(view, depth)
    points = points.permute(2, 0, 1) # (3, H, W)
    
    # Calcular gradientes usando padding para mantener el tamaño
    grad_y = torch.nn.functional.pad(points[:, 2:, :] - points[:, :-2, :], (0, 0, 1, 1))
    grad_x = torch.nn.functional.pad(points[:, :, 2:] - points[:, :, :-2], (1, 1, 0, 0))
    
    # Producto cruz para obtener la normal
    normal_map = torch.cross(grad_x, grad_y, dim=0)
    normal_map = torch.nn.functional.normalize(normal_map, dim=0)
    
    return normal_map.permute(1, 2, 0) # (H, W, 3)
