import os
import cv2
import numpy as np

def merge_images(input_folder, output_folder, patch_size):
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

        # 读取切片图像
        patch = cv2.imread(os.path.join(input_folder, file))

        # 移除填充
        pad_height = patch_size - patch.shape[0]
        pad_width = patch_size - patch.shape[1]
        unpadded_patch = patch[pad_height // 2: patch_size - pad_height // 2,
                               pad_width // 2: patch_size - pad_width // 2]

        # 将切片图像添加到原始图像的对应位置
        if image_name not in image_dict:
            image_dict[image_name] = np.zeros((patch_row + 1, patch_col + 1, patch_size, patch_size, 3), dtype=np.uint8)
        image_dict[image_name][patch_row, patch_col] = unpadded_patch

    # 合并切片图像并保存重构的原始图像
    for image_name, patches in image_dict.items():
        # 计算原始图像的尺寸
        image_shape = (patches.shape[0] * patch_size, patches.shape[1] * patch_size, 3)

        # 创建空白的原始图像
        merged_image = np.zeros(image_shape, dtype=np.uint8)

        # 将切片图像放入原始图像的对应位置
        for h in range(patches.shape[0]):
            for w in range(patches.shape[1]):
                merged_image[h * patch_size: (h + 1) * patch_size,
                             w * patch_size: (w + 1) * patch_size] = patches[h, w]

        # 保存重构的原始图像
        output_filename = image_name + '.png'
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, merged_image)


# 使用示例
input_folder = '../runs/Base_50_pred/lol/adapter/weights/epoch200'
output_folder = '../runs/Base_50_pred/lol/adapter/weights/epoch200_re'
patch_size = 224

merge_images(input_folder, output_folder, patch_size)
