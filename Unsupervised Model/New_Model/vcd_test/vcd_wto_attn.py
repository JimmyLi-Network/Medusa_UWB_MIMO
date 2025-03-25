import torch
# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
import pdb
import numpy as np
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

seq_len = 800
bs = 20

class VCD(nn.Module):


    def __init__(self,
                 in_channels: int,
                 z_dim: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int,
                 p: float,
                 **kwargs) -> None:
        super(VCD, self).__init__()

    
        self.z_dim = z_dim


        self.encoder = nn.GRU(in_channels, encoder_hidden_size, batch_first=True, num_layers=1, bias=True, bidirectional=True)
        

        self.decoder = nn.GRU(decoder_hidden_size, decoder_hidden_size, batch_first=True, num_layers=1, bias=True)
        
        self.z_mu = nn.Linear(encoder_hidden_size*2, z_dim)
        self.z_var = nn.Linear(encoder_hidden_size*2, z_dim)
        
        self.decoder_input_z = nn.Linear(z_dim, decoder_hidden_size)

        self.linear1 = nn.Linear(decoder_hidden_size, 32)
        self.linear2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p)

    def encode(self, input: Tensor) -> Tensor:

        input = input.permute(0,2,1).contiguous()
        #input(B, L, in_channels)
    
        en_out, en_hn = self.encoder(input)        
        en_out = self.dropout((en_out))
        z_mu = self.z_mu(en_out)
        z_var = self.z_var(en_out)
        
        return en_out, z_mu, z_var
    
    def decode(self, de_in: Tensor) -> Tensor:

        de_out, de_hn = self.decoder(de_in)
        de_out = self.dropout(de_out)
        result = self.linear1(de_out)
        result = F.relu(result)
        result = self.linear2(result)
        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def reparameterize_test(self, mu: Tensor, logvar: Tensor) -> Tensor:

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu

    def forward(self, input: Tensor, train=True, **kwargs) -> List[Tensor]:
        
        # input(B, in_channels ,L)
        en_out, z_mu, z_var = self.encode(input)
        #z_mu(B, z_dim), z_var(B, z_dim)
        if train == True:
            z = self.reparameterize(z_mu, z_var)
        else:
            z = self.reparameterize_test(z_mu, z_var)
        de_in_z = self.decoder_input_z(z)

        return self.decode(de_in_z), z_mu, z_var
