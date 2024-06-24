# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import  numpy as np
from typing import List
from mne.filter import resample

from src.models.components.modules import PixelNorm, ConvBlock, PrintLayer, WS

class CriticBlock(nn.Module):
    r"""
    Description
    ----------
    Single stage for the progressive growing critic.
    Each block consists of two possible modes:
    
    If first is True, the input data is passed through the in_sequence
    and then through the convolution_sequence.
    
    If first is False, the input data is passed through the 
    convolution_sequence only.

    To iniziialize the block, the following parameters are needed:

    Parameters:
    ----------
    intermediate_sequence : nn.Sequential
        Sequence of modules that process stage
    in_sequence : nn.Sequential
        Sequence of modules that is applied if stage is the current input

    """

    def __init__(self, intermediate_sequence:nn.Sequential, in_sequence:nn.Sequential) -> None:
        super().__init__()
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = in_sequence    

    def forward(self, x, first=False, **kwargs) -> torch.Tensor:
        if first:
            x = self.in_sequence(x, **kwargs)
        out = self.intermediate_sequence(x, **kwargs)
        return out
    
    def stage_requires_grad(self, requires_grad:bool) -> None:
        for module in [self.intermediate_sequence, self.in_sequence]:
            for param in module.parameters():
                param.requires_grad = requires_grad
    
class Critic(nn.Module):
    r"""
    Description
    ----------
    Critic module for implementing progressive GANs

    Parameters:
    ----------
    n_samples : int
        Number of timepoints in the input data
    n_classes : int
        Number of classes
    fading : bool
        If fading is used
    freeze : bool
        If parameters of past stages are frozen
    
    """

    def __init__(self,
                 n_samples:int,
                 n_classes:int,
                 fading:bool=False,
                 freeze:bool=False,
                 **kwargs
                 ) -> None:
         
        super().__init__()

        self.blocks:List[CriticBlock] = list()

        self.n_samples = n_samples
        self.fading = fading
        self.freeze = freeze
        self.label_embedding = nn.Embedding(n_classes, n_samples)
        self.alpha = 0

    def set_stage(self, cur_stage:int) -> None:
        # Each time a new stage is set, some parameters need to be updated:
        ## 1. The current stage:
        self.cur_stage = cur_stage

        # Internal stage variable.
        self._stage = len(self.blocks) - self.cur_stage 
        
        ## 2. Freeze or unfreeze parameters:
        if self.freeze:
            # Freeze all blocks
            for block in self.blocks:
                block.stage_requires_grad(False)
            # Unfreeze current block
            self.blocks[self._stage].stage_requires_grad(True)
        else:
            # Unfreeze all stages
            for block in self.blocks:
                block.stage_requires_grad(True)

        # In the first stage we do not need fading and therefore override alpha to 1
        if self.cur_stage == 1:
            self.alpha = 1
        else:
            self.alpha = 0

    def forward(self, x:torch.Tensor, y:torch.Tensor, **kwargs):
        
        embedding:torch.Tensor = self.label_embedding(y).view(y.shape[0], 1, self.n_samples)
        embedding = self.resample(embedding, x.shape[-1])
        
        x = torch.cat([x, embedding], 1) # batch_size x (n_channels + 1) x n_time 

        for i in range(self._stage, len(self.blocks)):
            first = (i == self._stage)
            if first and self.fading and self.alpha < 1:
                # if this is the first stage, fading is used and alpha < 1
                # we take the current input, downsample it for the next block
                # match dimensions by using in_sequence and interpolate with
                # the current bock output with the downsampled input. 
                x_ = self.resample(x, x.shape[-1] // 2)
                x_ = self.blocks[i-1].in_sequence(x_, **kwargs)

                # pass x through new (current) block
                x = self.blocks[i](x, first=first,  **kwargs)
                # interpolate X_ and X
                x = self.alpha * x + (1 - self.alpha) * x_  
            else:
                x = self.blocks[i](x,  first=first, **kwargs)
        return x
    
    def resample(self, x:torch.Tensor, out_size:int):
        """
        rescale input. Using bicubic interpolation.

        Parameters
        ----------
        X : tensor
            Input data
        
        out_size : tuple

        Returns
        -------
        X : tensor
            resampled data
        """
        size = (x.shape[-2], out_size)
        x = torch.unsqueeze(x, 1)
        x = nn.functional.interpolate(x, size=size, mode='bicubic')
        x = torch.squeeze(x, 1)

        return x
    
    def build(self, n_filter:int, n_samples:int, n_stages:int, n_channels:int, kernel_size=3) -> List[CriticBlock]:
        
        r"""
        This function builds the critic blocks for the progressive growing GAN.
        
        Arguments:
        ----------
        n_filter : int
            Number of filters in the convolutional layers
        n_samples : int
            Number of timepoints in the input data
        n_stages : int
            Number of stages in the critic
        n_channels : int
            Number of channels in the input data
        kernel_size : int
            Size of the convolutional kernel
        """

        raise NotImplementedError("This function needs to be implemented in the child class")

    def description(self) -> None:
        r"""
        When implementing a new critic, this function can be overwritten to provide a description of the model.
        """
        print(
        r"""
        No Description Set    
        """
        )