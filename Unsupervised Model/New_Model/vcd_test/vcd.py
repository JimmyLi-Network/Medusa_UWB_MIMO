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
                 q_size: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int,
                 p: float,
                 **kwargs) -> None:
        super(VCD, self).__init__()

    
        self.z_dim = z_dim
        
        self.query_ = nn.Sequential(nn.Linear(in_channels, q_size),
                                    nn.ReLU())
        self.key_ = nn.Sequential(nn.Linear(in_channels, q_size),
                                    nn.ReLU())
        self.value_ = nn.Sequential(nn.Linear(in_channels, q_size),
                                    nn.ReLU())

        self.encoder = nn.GRU(q_size*2, encoder_hidden_size, batch_first=True, num_layers=1, bias=True, bidirectional=True)
        

        self.decoder = nn.GRU(decoder_hidden_size, decoder_hidden_size, batch_first=True, num_layers=1, bias=True)
        
        self.z_mu = nn.Linear(encoder_hidden_size*2, z_dim)
        self.z_var = nn.Linear(encoder_hidden_size*2, z_dim)
        
        self.decoder_input_z = nn.Linear(z_dim, decoder_hidden_size)

        self.linear1 = nn.Linear(decoder_hidden_size, 32)
        self.linear2 = nn.Linear(32, 1)

        self.attention = ScaledDotProductAttention(q_size)
        
        self.dropout = nn.Dropout(p)

    def encode(self, input: Tensor) -> Tensor:

        input = input.permute(0,2,1).contiguous()
        #input(B, L, in_channels)
        query = self.query_(input)
        value = self.value_(input)
        key = self.key_(input)
        context, attn = self.attention(query,key, value)
        encode_input = torch.cat((query,context),2).permute(0,1,2).contiguous()
        en_out, en_hn = self.encoder(encode_input)        
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

    def forward(self, input_i: Tensor, input_q: Tensor, train=True, **kwargs) -> List[Tensor]:
        
        # input(B, in_channels ,L)
        # for rf data (i + q)
        input = torch.cat((input_i, input_q),1)
        en_out, z_mu, z_var = self.encode(input)
        #z_mu(B, z_dim), z_var(B, z_dim)
        if train == True:
            z = self.reparameterize(z_mu, z_var)
        else:
            z = self.reparameterize_test(z_mu, z_var)
        de_in_z = self.decoder_input_z(z)

        return self.decode(de_in_z), z_mu, z_var



class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2))/ self.sqrt_dim

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn