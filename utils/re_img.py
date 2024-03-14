import os
import cv2
import numpy as np


def reconstruct_image(input_folder, output_folder, target_size):
    # 创建保存文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有切片图像文件
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # 存储原始图像的字典
    image_dict = {}

    for file in files:
        # 解析切片图像文件名
        filename = os.path.splitext(file)[0]
        filename_parts = filename.split('_')
        image_name = filename_parts[0] + "_" + filename_parts[4]  # 提取图像名称
        patch_row = int(filename_parts[-4])  # 提取切片行号
        patch_col = int(filename_parts[-3])  # 提取切片列号
        patch_size = int(filename_parts[-2])  # 提取切片大小

        # 读取切片图像
        patch = cv2.imread(os.path.join(input_folder, file))

        h, w, _ = patch.shape
        # 计算填充大小
        pad_height = h - patch_size
        pad_width = w - patch_size

        # 去除填充
        unpadded_patch = patch[pad_height // 2: h - pad_height // 2, pad_width // 2: w - pad_width // 2]

        # 更新原始图像的尺寸
        if image_name not in image_dict:
            # 如果是第一次遇到该原图，创建一个空白的切片图像列表
            image_dict[image_name] = []
        image_dict[image_name].append((patch_row, patch_col, unpadded_patch))

        # 根据原图的切片图像列表，进行拼接
    for image_name, patches in image_dict.items():
        # 根据切片位置排序
        patches.sort()

        # 计算重构图像的尺寸
        num_rows = patches[-1][0] + 1
        num_cols = patches[-1][1] + 1
        height = num_rows * patch_size
        width = num_cols * patch_size

        # 创建一个空白的重构图像
        image_dict[image_name] = np.zeros((height, width, 3), dtype=np.uint8)

        for patch_row, patch_col, patch in patches:
            # 计算切片在重构图像中的位置
            top = patch_row * patch_size
            left = patch_col * patch_size
            bottom = top + patch_size
            right = left + patch_size

            # 将切片放置在重构图像中的对应位置
            image_dict[image_name][top:bottom, left:right] = patch

    # 保存重构图像
    for image_name, image in image_dict.items():
        output_filename = image_name + '.png'
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)

# 使用示例
input_folder = '../runs/CNN_pred_h8/lol/test2_vit5-12/weights/epoch300'
output_folder = '../runs/CNN_pred_h8/lol/test2_vit5-12/weights/epoch300_re'
patch_size = 224

reconstruct_image(input_folder, output_folder, patch_size)


