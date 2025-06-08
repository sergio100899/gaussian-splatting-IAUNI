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
