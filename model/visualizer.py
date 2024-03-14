from bytecode import Bytecode, Instr
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

class get_local(object):
    # 类变量，用于缓存捕获的值
    cache = {}
    # 类变量，用于指示是否激活捕获
    is_activate = False

    def __init__(self, varname):
        # 初始化函数，接收要捕获的变量名
        self.varname = varname

    def __call__(self, func):
        # 装饰器的调用方法，在函数调用时执行

        # 如果未激活捕获，直接返回原始函数
        if not type(self).is_activate:
            return func

        # 初始化函数的缓存为一个空列表
        type(self).cache[func.__qualname__] = []

        # 获取函数的字节码对象
        c = Bytecode.from_code(func.__code__)

        # 额外的字节码指令，用于捕获变量的值
        extra_code = [
                         Instr('STORE_FAST', '_res'),
                         Instr('LOAD_FAST', self.varname),
                         Instr('STORE_FAST', '_value'),
                         Instr('LOAD_FAST', '_res'),
                         Instr('LOAD_FAST', '_value'),
                         Instr('BUILD_TUPLE', 2),
                         Instr('STORE_FAST', '_result_tuple'),
                         Instr('LOAD_FAST', '_result_tuple'),
                     ]

        # 在原始字节码的最后插入额外的字节码指令
        c[-1:-1] = extra_code

        # 更新函数的字节码
        func.__code__ = c.to_code()

        # 定义一个新的函数包装器，用于执行原始函数，并将捕获的值添加到缓存中
        def wrapper(*args, **kwargs):
            res, values = func(*args, **kwargs)
            # 将捕获的值添加到缓存中s
            type(self).cache[func.__qualname__].append(values.detach().cpu().numpy())
            return res

        # 返回新的函数包装器
        return wrapper

    @classmethod
    def clear(cls):
        # 类方法，用于清空缓存
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        # 类方法，用于激活捕获
        cls.is_activate = True


def grid_show(to_shows, cols):
    """
    将图像以网格形式显示。

    参数：
    - to_shows: 包含图像和标题的列表
    - cols: 列数
    """
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()


def visualize_head(att_map):
    """
    可视化单个注意力头的热力图。

    参数：
    - att_map: 注意力热力图
    """
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def visualize_heads(att_map, cols):
    """
    可视化多个注意力头的平均热力图。

    参数：
    - att_map: 注意力热力图的集合
    - cols: 列数
    """
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def gray2rgb(image):
    """
    将灰度图转换为RGB格式。

    参数：
    - image: 输入的灰度图
    返回：
    - 转换后的RGB图像
    """
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    """
    在图像中添加分类（CLS）标记。

    参数：
    - image: 输入的图像
    - mask: 分类标记的掩模
    - cls_weight: 分类标记的权重
    - grid_size: 网格大小
    返回：
    - 添加分类标记后的图像、掩模和元掩模
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    """
    可视化包含分类（CLS）标记的注意力图。

    参数：
    - att_map: 注意力图
    - grid_index: 网格索引
    - image: 输入的图像
    - grid_size: 网格大小
    - alpha: 透明度
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    """
    可视化注意力图中的单个网格。

    参数：
    - att_map: 注意力图
    - grid_index: 网格索引
    - image: 输入的图像
    - grid_size: 网格大小
    - alpha: 透明度
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()


def highlight_grid(image, grid_indexes, grid_size=14):
    """
    在图像上突出显示指定网格。

    参数：
    - image: 输入的图像
    - grid_indexes: 要突出显示的网格索引列表
    - grid_size: 网格大小
    返回：
    - 突出显示后的图像
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image