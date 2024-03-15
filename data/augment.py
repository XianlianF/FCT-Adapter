# 数据增广操作
import random
import torch

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

def augment_train(clean_img, noisy_img, patch_size):
    w, h = clean_img.size
    ps = patch_size
    padw = ps - w if w < ps else 0
    padh = ps - h if h < ps else 0

    # Reflect Pad in case image is smaller than patch_size
    if padw != 0 or padh != 0:
        clean_img = TF.pad(clean_img, (0, 0, padw, padh), padding_mode='reflect')
        noisy_img = TF.pad(noisy_img, (0, 0, padw, padh), padding_mode='reflect')

    clean_img = TF.to_tensor(clean_img)
    noisy_img = TF.to_tensor(noisy_img)

    hh, ww = noisy_img.shape[1], noisy_img.shape[2]

    rr = random.randint(0, hh - ps)
    cc = random.randint(0, ww - ps)
    aug = random.randint(0, 8)

    # Crop patch
    clean_img = clean_img[:, rr:rr + ps, cc:cc + ps]
    noisy_img = noisy_img[:, rr:rr + ps, cc:cc + ps]

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # clean_img = normalize(clean_img)
    # noisy_img = normalize(noisy_img)

    # Data Augmentations
    if aug == 1:
        clean_img = clean_img.flip(1)
        noisy_img = noisy_img.flip(1)
    elif aug == 2:
        clean_img = clean_img.flip(2)
        noisy_img = noisy_img.flip(2)
    elif aug == 3:
        clean_img = torch.rot90(clean_img, dims=(1, 2))
        noisy_img = torch.rot90(noisy_img, dims=(1, 2))
    elif aug == 4:
        clean_img = torch.rot90(clean_img, dims=(1, 2), k=2)
        noisy_img = torch.rot90(noisy_img, dims=(1, 2), k=2)
    elif aug == 5:
        clean_img = torch.rot90(clean_img, dims=(1, 2), k=3)
        noisy_img = torch.rot90(noisy_img, dims=(1, 2), k=3)
    elif aug == 6:
        clean_img = torch.rot90(clean_img.flip(1), dims=(1, 2))
        noisy_img = torch.rot90(noisy_img.flip(1), dims=(1, 2))
    elif aug == 7:
        clean_img = torch.rot90(clean_img.flip(2), dims=(1, 2))
        noisy_img = torch.rot90(noisy_img.flip(2), dims=(1, 2))

    return clean_img, noisy_img


def augment_val(clean_img, noisy_img, patch_size):
    # w, h = clean_img.size
    # ps = patch_size
    # H, W = ((h + ps) // ps) * ps, ((w + ps) // ps) * ps
    # padh = H - h if h % ps != 0 else 0
    # padw = W - w if w % ps != 0 else 0
    # clean_img = TF.pad(clean_img, (padw / 2, padh / 2, padw / 2, padh / 2), padding_mode='reflect')
    # noisy_img = TF.pad(noisy_img, (padw / 2, padh / 2, padw / 2, padh / 2), padding_mode='reflect')
    clean_img = TF.to_tensor(clean_img)
    noisy_img = TF.to_tensor(noisy_img)

    return clean_img, noisy_img
