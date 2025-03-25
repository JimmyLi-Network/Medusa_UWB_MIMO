import torch
import numpy as np
from dataset import *
from torch import optim

from vcd import VCD
from torch.utils.tensorboard import SummaryWriter
import pdb
from scipy.io import savemat
import time

PATH = './model.pth'
model = VCD(12, 8, 64, 64, 128, 0.2)
model.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))

testset = MyDataset('../dataset/test_x.npy','../dataset/test_y.npy')#, transform1=transform1, transform2 = transform2)
dataloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=2)

res = np.empty((0, 800), float)
tgt = np.empty((0, 800), float)

def test(model, dataloader, res, tgt):
    for i, data in enumerate(dataloader, 0):
        print(i)
        inputs, labels = data 
        outputs = model(inputs, False)
        res = np.append(res, outputs[0].squeeze().detach().numpy(), axis=0)
        tgt = np.append(tgt, labels.numpy(), axis=0)
        

    mdic = {"res":res}
    savemat("estimated_source.mat", mdic)

    mdic = {"tgt":tgt}
    savemat("target_source.mat", mdic)
    
    return res, tgt


if __name__ == "__main__":

    res, tgt = test(model, dataloader, res, tgt)

    
