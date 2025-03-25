import torch
import numpy as np
from dataset import *
from torch import optim
from torch.nn import MSELoss
from torch.nn import functional as F
from vcd import VCD
import pdb
import math
import time

torch.manual_seed(0)

device= torch.device("cpu")

bs = 20


def loss_function(recons, z_mu, z_log_var, gt, kld_weight) -> dict:
   
    recons_loss_ = MSELoss()
    recons_loss = recons_loss_(recons.view(bs,-1), gt.view(bs,-1))

    z_mu = z_mu.mean(dim=1)
    z_log_var = z_log_var.mean(dim=1)
    
    kld_loss =  (-0.5*(1 + z_log_var - z_mu**2 - torch.exp(z_log_var)).mean(dim=1)).mean(dim=0)
    kld_loss = kld_loss*kld_weight
    loss = recons_loss + kld_loss
    return loss, recons_loss, kld_loss


def train(model, dataloader, save_name, kl_weight):
    #early stop for kld if need
    kldloss_stop = 0.02
    epoch_start_time = time.time()

    for epoch in range(1): 
        running_loss = 0
        running_recons_loss = 0
        running_kldloss = 0
        my_lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=my_lr)
        steps = 10
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        for i, data in enumerate(dataloader, 0):
            # get the inputs; 
            # get the inputs; 
            inputs_i, inputs_q, labels = data
    

            # zero the parameter gradients
            optimizer.zero_grad()
            inputs_i = inputs_i.to(device)
            inputs_q = inputs_q.to(device)
            labels = labels.to(device)
            outputs = model(inputs_i,inputs_q)

            gamma = kl_weight
            loss,recons_loss,kldloss = loss_function(recons = outputs[0], z_mu = outputs[1], z_log_var = outputs[2], gt = labels, kld_weight = gamma)
            #if kldloss < kldloss_stop:
            #    recons_loss.backward()
            #else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-5)
            optimizer.step()
            
            running_loss += loss.detach().item()
            running_recons_loss += recons_loss.detach().item()
            running_kldloss += kldloss.detach().item()
            if (i+1) % 5 == 0:    
               scheduler.step()

        # print statistics
        print('[%d, %5d] loss: %.4f  running_recons_loss: %.4f  running_kldloss: %.4f  gamma: %.4f' % (epoch + 1, i + 1, running_loss, running_recons_loss, running_kldloss, gamma))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), './'+save_name+'.pth')
            print('time elapsed: ', (time.time() - epoch_start_time)//3600, 'h ',(time.time() - epoch_start_time)%3600)
    
    torch.save(model.state_dict(), './'+save_name+'.pth')
    print('Finished Training')

if __name__ == "__main__":

    #in_channels: int, z_dim: int, q_size: int, encoder_hidden_size: int, decoder_hidden_size: int, p: float,
    model = VCD(10, 8, 64, 64, 128, 0.2).to(device)
    #model.load_state_dict(torch.load('./model.pth'))
    #model.to(device)
    #model.train()

    trainset = MyDataset('./data_debug.mat')#, transform1=transform1, transform2 = transform2)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
    train(model, dataloader,'model',0.1)
