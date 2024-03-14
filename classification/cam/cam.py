from utils.torch_utils import load_match_dict, load_dict
from options import prepare_train_args
from utils.classification_utils import *

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
import torch
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import cv2
from timm.models import create_model
from model.FCT_Adapter.Base_50_adapter import Base_50_adapter, set_adapter
from vit_rollout import VITAttentionRollout
from vit_explain import show_mask_on_image
import os

def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
    return model


def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    tensor = tensor.to('cpu')
    np_arr = tensor.detach().numpy()  # [0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    # np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr


# 利用cam做vit的注意力需要的transform
def reshape_transform(tensor, height=14, width=14):
    # 去掉类别标记
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# 输出为元组的操作
def cam_transform(tuple):
    return tuple[1]


args = prepare_train_args()  # 初始化命令行参数
set_seed(args.seed)

gpus = args.gpus
device = torch.device(gpus)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

pt = 'dmlab'

with open('/media/dl_shouan/DATA/fxl/configs/' +pt + '.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

img_name = '000131.jpg'

path = r'/media/dl_shouan/DATA/fxl/data/vtab-1k/' + pt + '/images/test/' + img_name

# 输入图像转化
img = cv2.imread(path)
input_tensor = transform(img)
input_tensor = torch.unsqueeze(input_tensor, 0)
input_tensor = input_tensor.to(device)

'''FCT-Adapter'''
resnet50 = create_model('resnet50', num_classes=11221, drop_path_rate=0.1)
resnet50 = load_model_weights(resnet50, '/media/dl_shouan/DATA/fxl/pre_weight/resnet50_21k.pth')
resnet50.reset_classifier(config['class_num'])
vit_model = create_model('vit_base_patch16_224_in21k',
                         checkpoint_path='/media/dl_shouan/DATA/fxl/pre_weight/ViT-B_16.npz',
                         drop_path_rate=0.1)
vit_model.reset_classifier(config['class_num'])
set_adapter(resnet50, "Convchor", f=8, h=8, r=1, s=config['scale'])
set_adapter(vit_model, "Transchor", f=8, h=8, r=1, s=config['scale'])

model = Base_50_adapter(cnn=resnet50, vit=vit_model, m=8, a=1, num_classes=config['class_num'], embed_dim=768)
load_match_dict(model, '/media/dl_shouan/DATA/fxl/runs/FCT-Adapter/vtab/Resnet50_ViT-B_adapter/weights/' + pt + '.pth')

'''resnet50'''
model_r = create_model('resnet50', num_classes=config['class_num'])
model_r = load_model_weights(model_r, '/media/dl_shouan/DATA/fxl/pre_weight/resnet50_21k.pth')

'''resnet50_vtab权重'''
model_r_vtab = create_model('resnet50', num_classes=config['class_num'])
load_match_dict(model_r_vtab, '/media/dl_shouan/DATA/fxl/runs/Resnet50/vtab/full/weights/' + pt + '.pth')

'''vit-b'''
model_v = create_model('vit_base_patch16_224_in21k',
                         checkpoint_path='/media/dl_shouan/DATA/fxl/pre_weight/ViT-B_16.npz',
                         drop_path_rate=0.1)

'''vit-b_vtab权重'''
model_v_vtab = create_model('vit_base_patch16_224_in21k', num_classes=config['class_num'], drop_path_rate=0.1)
load_match_dict(model_v_vtab, '/media/dl_shouan/DATA/fxl/runs/VIT-B/full/weights/' + pt + '.pth')

model.to(device)
model_r.to(device)
model_v.to(device)
model_r_vtab.to(device)
model_v_vtab.to(device)

# for key, value in model.state_dict().items():
#     print(key)

for name, module in model.named_modules():
    print(name)


def visualize_gradcam(model, input_tensor, save_path, target_layers=None):
    """
    使用GradCAM对模型的特定层进行可视化。

    Args:
        model: 模型对象
        input_tensor: 输入图像的张量表示
        save_path: 结果图像保存路径
        target_layers: 可视化位置，默认为None

    Returns:
        None
    """

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 创建GradCAM对象
    with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
        # 获取热力图结果
        grayscale_cams = cam(input_tensor=input_tensor)
        for grayscale_cam, tensor in zip(grayscale_cams, input_tensor):
            # 将热力图结果与原图进行融合
            rgb_img = tensor2img(tensor)
            heatmap_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # 保存结果图像
            plt.imsave(save_path, heatmap_img, cmap='hot')
            print(f"Image saved at: {save_path}")


def visualize_attention_rollout(model, input_tensor, img, save_path, resize=False):
    """
    对注意力分布进行可视化，将注意力分布叠加到原图上，并保存结果图像。

    Args:
        model: 模型对象
        input_tensor: 输入图像的张量表示
        img: 原始图像
        save_path: 结果图像保存路径
        resize: 是否调整图像大小，默认为False

    Returns:
        None
    """

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 创建VITAttentionRollout对象
    grad_rollout = VITAttentionRollout(model, attention_layer_name='attn_drop', discard_ratio=0.8, head_fusion='mean')
    # 获取注意力分布
    mask = grad_rollout(input_tensor)

    # 图片大小不变
    if not resize:
        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    else:
        # 将图片大小调整为224x224
        np_img = np.array(img)[:, :, ::-1]
        np_img = cv2.resize(np_img, (224, 224))
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))

    # 将注意力分布叠加到原图上
    mask = show_mask_on_image(np_img, mask)

    # 保存结果图像
    b, g, r = cv2.split(mask)
    mask = cv2.merge((r, g, b))
    plt.imsave(save_path, mask)
    print(f"Image saved at: {save_path}")


# target_layers = [model.layer4[0].act2]
target_layers = [model.layer4[-1].act2]
target_layers_r = [model_r.layer4[-1].act3]
target_layers_r_vtab = [model_r_vtab.layer4[-1].act3]
visualize_attention_rollout(model, input_tensor, img, '../image/adapter/ViT/' + pt + '/' + img_name, True)
visualize_attention_rollout(model_v, input_tensor, img, '../image/vit/pre/' + pt + '/' + img_name, True)
visualize_attention_rollout(model_v_vtab, input_tensor, img, '../image/vit/vtab/' + pt + '/' + img_name, True)
visualize_gradcam(model, input_tensor, '../image/adapter/CNN/' + pt + '/' + img_name, target_layers)
visualize_gradcam(model_r, input_tensor, '../image/resnet/pre/' + pt + '/' + img_name, target_layers_r)
visualize_gradcam(model_r_vtab, input_tensor, '../image/resnet/vtab/' + pt + '/' + img_name, target_layers_r_vtab)
