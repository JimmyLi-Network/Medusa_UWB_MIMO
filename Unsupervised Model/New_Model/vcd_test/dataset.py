import scipy.io as io
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms, utils

class MyDataset(Dataset):
    def __init__(self, mat_path): #, transform1=None, transform2=None):
        data = io.loadmat(mat_path)
        self.images_i = torch.from_numpy(data['rf_data_i'].astype(np.float32))
        self.images_q = torch.from_numpy(data['rf_data_q'].astype(np.float32))
        self.targets = torch.from_numpy(data['bpm_data'].astype(np.float32))
        # self.transform1= transform1
        # self.transform2= transform2

    def __getitem__(self, index):
        x_i = self.images_i[index]
        x_q = self.images_q[index]
        y = self.targets[index]
        # x.unsqueeze_(0)
        # y.unsqueeze_(0)
        return x_i, x_q, y #self.transform1(x), self.transform2(y)

    def __len__(self):
        return len(self.images_i)