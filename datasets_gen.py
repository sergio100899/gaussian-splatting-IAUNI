from utils.loss_utils import sobel_edges, depth_inference  # noqa

import os
import argparse
import cv2
import torch
import numpy as np


def aplicar_filtro(img):
    # Convertir la imagen de OpenCV a un tensor de PyTorch
    img_tensor = (
        torch.from_numpy(img).float().permute(2, 0, 1)
    )  # Reordenar dimensiones si es necesario
    return img_tensor


def procesar_imagenes(input_folder, output_folder, filtro):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

        if img is not None:
            filtered_img = aplicar_filtro(img)
            output_path = os.path.join(output_folder, filename)
            if filtro == "depths":
                depth_tensor = depth_inference(filtered_img).squeeze(0).to("cuda")
                depth_np_norm = (depth_tensor - depth_tensor.min()) / (
                    depth_tensor.max() - depth_tensor.min()
                )
                cv2.imwrite(
                    output_path, (depth_np_norm.cpu().numpy() * 255).astype(np.uint8)
                )
            elif filtro == "normals":
                depth_tensor = (
                    depth_inference(filtered_img).squeeze(0).to("cuda")
                )  # Obtener la profundidad

                # Calcular el mapa de normales
                zy, zx = torch.gradient(depth_tensor.float(), dim=[0, 1])
                normal_map = torch.stack(
                    [-zx, -zy, torch.ones_like(depth_tensor)], dim=2
                )

                # Normalización
                norm = torch.linalg.norm(normal_map, dim=2, keepdim=True) + 1e-8
                normal_map = normal_map / norm

                print(normal_map.ndim)

                # Convertir a imagen RGB
                normal_rgb = (
                    ((normal_map + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)
                )
                normal_bgr = cv2.cvtColor(normal_rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)

                cv2.imwrite(output_path, normal_bgr)

            elif filtro == "edges":
                edges_tensor = sobel_edges(filtered_img)

                edges_np = edges_tensor.squeeze().detach().cpu().numpy()
                edges_np = (edges_np * 255).clip(0, 255).astype(np.uint8)

                cv2.imwrite(output_path, edges_np)
            print(f"Procesando imagen {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aplicar filtros a imágenes en una carpeta."
    )
    parser.add_argument(
        "input_folder", type=str, help="Carpeta de imágenes originales."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Carpeta donde se guardarán las imágenes filtradas.",
    )
    parser.add_argument(
        "filtro",
        type=str,
        choices=["depths", "normals", "edges"],
        help="Filtro a aplicar (gris, invertir, bordes).",
    )

    args = parser.parse_args()
    procesar_imagenes(args.input_folder, args.output_folder, args.filtro)

    # sobel_tensor = load_image_as_tensor_pil(
    #     args.input_folder + "/0001.png", grayscale=True
    # )
