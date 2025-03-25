import torch
import numpy as np
from dataset import *
from torch import optim

from vcd import VCD
from torch.utils.tensorboard import SummaryWriter
import pdb
from scipy.io import savemat
import time

#device= torch.device("cpu")
PATH = './model.pth'
model = VCD(10, 8, 64, 64, 128, 0.2)#.to(device)
model.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))

testset = MyDataset('./data_debug.mat')#, transform1=transform1, transform2 = transform2)
dataloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=2)

res = np.empty((0, 800), float)
tgt = np.empty((0, 800), float)

res = np.empty((0, 800), float)
tgt = np.empty((0, 800), float)
def test(model, dataloader, res, tgt):
    for i, data in enumerate(dataloader, 0):
        # pdb.set_trace()
        print(i)
        inputs_i, inputs_q, labels = data

        # t = time.time()
        outputs = model(inputs_i, inputs_q, False)
        # elapsed = time.time()-t
        # print(elapsed)
        res = np.append(res, outputs[0].squeeze().detach().numpy(), axis=0)

    mdic = {"ebpm":res}
    savemat("estimated.mat", mdic)


if __name__ == "__main__":
    test(model, dataloader, res, tgt)