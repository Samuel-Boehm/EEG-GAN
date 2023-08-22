# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import torch.nn as nn
import torch
from GAN.Model.Modules import PixelNorm, ConvBlockStage, PrintLayer, WS


class spectralCriticStage(nn.Module):
    """
    Simple version of the spectral Critic. 
    The spectral vector is passed through a single linear layer with leaky relu activation.

    Args:
        in_size (int): size of the input vector
        out_size (int): size of the output vector    
    """

    def __init__(self, in_size, out_size):
        super(spectralCriticStage, self).__init__()
        print('Linear sizes')
        print(in_size, out_size)
        self.lin = nn.Linear(in_size, out_size)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, **kwargs):
        x = self.leaky(self.lin(x))
        return x


class spectralCritic(nn.Module):
        
    def __init__(self, blocks, stage=1, fading=False):
        super(spectralCritic, self).__init__()
        self.blocks  = nn.ModuleList(blocks)
        self.set_stage(stage)
        self.fading = fading
        self.alpha = 0
    
    def set_stage(self, stage):
        self.cur_stage = stage
        self._stage = len(self.blocks) - self.cur_stage # Internal stage variable. Differs from GAN stage (cur_stage).
        # In the first stage we do not need fading and therefore set alpha to 1
        if self.cur_stage == 1:
            self.alpha = 1
        else:
            self.alpha = 0
    
    def forward(self, x, y):

        x = self.spectral_vector(x)
        x = self.blocks[self._stage](x)
        return x
    
    def spectral_vector(self, x):
        """
        Calculates the fast fourier transform of the input vector and returns the mean of the absolute values of the fft (aka Aptitudes).

        Assumes dimensions to be batch_size x channels x time
        """
        fft = torch.fft.rfft(x)
        fft_abs = torch.abs(fft)
        fft_abs = fft_abs + 1E-8
        fft_abs = torch.log(fft_abs)
        fft_mean = fft_abs.mean(axis=(0, 1)).squeeze()
        return fft_mean

def build_sp_critic(n_filters, n_time, n_stages, n_channels, n_classes, fading):

    # Calculate the number of timepoints in the last layer
    # n_stages - 1 since we dont downsample after the last convolution
    n_time_last_stage = int(np.floor(n_time / 2 ** (n_stages - 1)))
    
    # Critic:
    blocks = []

    for stage in  range(n_stages):
        blocks.append(spectralCriticStage(int(((n_time / 2 ** stage) / 2 ) + 1), 1))

    return spectralCritic(blocks, fading=fading)

