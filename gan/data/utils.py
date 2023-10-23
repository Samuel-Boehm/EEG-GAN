# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch
import torch.nn as nn

def downsample_step(batch,):
    r"""
    Downsample a batch of data to half the sampling rate.
    
    Params:
        batch       : Batch of data to downsample shape (batch_size, n_channels, n_time)
    """
    # Downsample the dataset to 128 Hz
    x = torch.unsqueeze(batch, 0)
    x = nn.functional.interpolate(x, scale_factor=(1, 0.5), mode='bicubic')
    x = torch.squeeze(x, 0)
    return x
