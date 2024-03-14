import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


def compute_patch_size(image_shape, target_size):
    # 计算最佳切片大小
    max_patch_size = min(image_shape[0], image_shape[1], target_size)
    patch_size = max_patch_size
    while patch_size <= target_size:
        if image_shape[0] % patch_size == 0 and image_shape[1] % patch_size == 0:
            break
        patch_size -= 1
    return patch_size


def split_and_save_images(input_folder, output_folder, target_size):
    # 创建保存文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图片文件
    files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for file in files:
        # 读取图片
        image = cv2.imread(os.path.join(input_folder, file))

        # 计算切片大小
        patch_size = compute_patch_size(image.shape, target_size)

        # 切片
        for h in range(0, image.shape[0] // patch_size):
            for w in range(0, image.shape[1] // patch_size):
                top = h * patch_size
                left = w * patch_size
                bottom = top + patch_size
                right = left + patch_size

                patch = image[top:bottom, left:right]

                # 计算填充大小
                pad_height = target_size - patch.shape[0]
                pad_width = target_size - patch.shape[1]

                # 对图像进行填充
                padded_patch = cv2.copyMakeBorder(patch, pad_height // 2, pad_height - pad_height // 2,
                                                  pad_width // 2, pad_width - pad_width // 2, cv2.BORDER_REFLECT)

                # 保存切片图像
                filename = os.path.splitext(file)[0]
                output_filename = f"{filename}_{h}_{w}_{patch_size}.png"
                output_path = os.path.join(output_folder, output_filename)

                # 处理同名文件覆盖情况
                if os.path.exists(output_path):
                    os.remove(output_path)

                cv2.imwrite(output_path, padded_patch)


# 使用示例
input_folder = '../data/LOL/LOLdataset/val/input'
output_folder = '../data/LOL/LOLdataset/split_val/input'
patch_size = 224

split_and_save_images(input_folder, output_folder, patch_size)

