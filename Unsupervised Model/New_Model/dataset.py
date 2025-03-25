import scipy.io as io
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms, utils

class MyDataset(Dataset):
    def __init__(self, x_pth, y_pth):
        x_data = np.load(x_pth,allow_pickle=True)
        y_data = np.load(y_pth,allow_pickle=True)
        self.x = torch.from_numpy(x_data.astype(np.float32))
        self.y = torch.from_numpy(y_data.astype(np.float32))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y 

    def __len__(self):
        return len(self.y)