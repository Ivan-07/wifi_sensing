import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MH_CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='Phase', transform=None, input_size=224):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (Mag/Phase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.input_size = input_size
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder = glob.glob(root_dir + '/*/')
        self.data_list = [path.replace('\\', '/') for path in self.data_list]
        self.folder = [path.replace('\\', '/') for path in self.folder]
        self.category = {self.folder[i].split('/')[-2]: int(self.folder[i].split('/')[-2][2:]) for i in
                         range(len(self.folder))}
        1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]

        mean = np.mean(x, axis=(0, 1, 2))  # 计算均值
        std = np.std(x, axis=(0, 1, 2))  # 计算标准差

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        if self.transform:
            x = self.transform(x)


        x = x.float()

        return x, y


