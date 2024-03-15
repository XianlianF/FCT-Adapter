import os

import numpy as np
import torch

from utils.image_utils import is_png_file, load_img

from torch.utils.data import Dataset

class Lol_dataset(Dataset):
    def __init__(self, args, is_train):
        super(Lol_dataset).__init__()
        self.is_train = is_train
        self.args = args
        if is_train:
            self.data_list = args.train_list
        else:
            self.data_list = args.val_list

        clean_files = sorted(os.listdir(os.path.join(self.data_list, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(self.data_list, 'input')))

        self.clean_filenames = [os.path.join(self.data_list, 'gt', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(self.data_list, 'input', x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean, noisy = load_img(self.clean_filenames[tar_index], self.noisy_filenames[tar_index], self.is_train, self.args.split_size)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return noisy, clean, clean_filename, noisy_filename
