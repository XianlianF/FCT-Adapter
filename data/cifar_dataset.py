import torchvision
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    def __init__(self, args, is_train):
        super(CifarDataset).__init__()
        if is_train:
            self.data_list = torchvision.datasets.CIFAR10(root=args.train_list + "/cifar-10-python", train=True,
                                                          transform=torchvision.transforms.ToTensor(),
                                                          download=False)
        else:
            self.data_list = torchvision.datasets.CIFAR10(root=args.val_list + "/cifar-10-python", train=False,
                                                          transform=torchvision.transforms.ToTensor(),
                                                          download=False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, label = self.data_list[idx]
        return img, label
