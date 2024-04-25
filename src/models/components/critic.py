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
        super(CriticBlock, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = in_sequence    

    def forward(self, X, first=False, **kwargs) -> torch.Tensor:
        if first:
            X = self.in_sequence(X, **kwargs)
        out = self.intermediate_sequence(X, **kwargs)
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
    n_filter : int
        Number of filters in the convolutional layers
    n_samples : int
        Number of timepoints in the input data
    n_stages : int
        Number of stages in the critic
    n_channels : int
        Number of channels in the input data
    n_classes : int
        Number of classes
    current_stage : int
        Current stage of the critic
    fading : bool
        If fading is used
    freeze : bool
        If parameters of past stages are frozen
    
    """

    def __init__(self,
                 n_filter:int,
                 n_samples:int,
                 n_stages:int,
                 n_channels:int,
                 n_classes:int,
                 current_stage:int=1,
                 fading:bool=False,
                 freeze:bool=False,
                 **kwargs
                 ) -> None:   
         
        super(Critic, self).__init__()
        self.blocks:List[CriticBlock] = self.build(n_filter, n_samples, n_stages, n_channels)
        self.n_samples = n_samples
        self.fading = fading
        self.freeze = freeze
        self.label_embedding = nn.Embedding(n_classes, n_samples)
        self.alpha = 0
        self.set_stage(current_stage)


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

    def forward(self, X:torch.Tensor, y:torch.Tensor, **kwargs):
        
        embedding:torch.Tensor = self.label_embedding(y).view(y.shape[0], 1, self.n_samples)
        embedding = self.resample(embedding, X.shape[-1])
        
        X = torch.cat([X, embedding], 1) # batch_size x (n_channels + 1) x n_time 

        for i in range(self._stage, len(self.blocks)):
            first = (i == self._stage)
            if first and self.fading and self.alpha < 1:
                # if this is the first stage, fading is used and alpha < 1
                # we take the current input, downsample it for the next block
                # match dimensions by using in_sequence and interpolate with
                # the current bock output with the downsampled input. 
                X_ = self.resample(X, X.shape[-1] // 2)
                X_ = self.blocks[i-1].in_sequence(X_, **kwargs)

                # pass x through new (current) block
                X = self.blocks[i](X, first=first,  **kwargs)
                # interpolate X_ and X
                X = self.alpha * X + (1 - self.alpha) * X_  
            else:
                X = self.blocks[i](X,  first=first, **kwargs)
        return X
    
    def resample(self, X:torch.Tensor, out_size:int):
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
        size = (X.shape[-2], out_size)
        X = torch.unsqueeze(X, 1)
        X = nn.functional.interpolate(X, size=size, mode='bicubic')
        X = torch.squeeze(X, 1)

        return X
    
    def build(self, n_filter:int, n_samples:int, n_stages:int, n_channels:int) -> List[CriticBlock]:
        
        n_channels += 1 # Add one channel for embedding

        # Calculate the number of timepoints in the last layer
        # n_stages - 1 since we dont downsample after the last convolution
        n_time_last_stage = int(np.floor(n_samples / 2 ** (n_stages - 1)))
        
        # Critic:
        blocks = nn.ModuleList()

            
        critic_in = nn.Sequential(
            WS(nn.Conv1d(n_channels, n_filter, kernel_size=1, stride=1)), #WS()
            nn.LeakyReLU(0.2),
        )

        downsample = nn.Sequential(nn.ReflectionPad1d(1),
                                    WS(nn.Conv1d(n_filter, n_filter, kernel_size=4, stride=2)), #WS()
                                    nn.LeakyReLU(0.2))

        for stage in range(1, n_stages):
            stage_conv = nn.Sequential(
                        ConvBlock(n_filter, n_stages - stage, is_generator=False),
                        downsample)

            # In sequence is independent of stage
            blocks.append(CriticBlock(stage_conv, critic_in))


        
        final_conv = nn.Sequential(
            ConvBlock(n_filter, 0, is_generator=False),
            nn.Flatten(),
            nn.Linear(n_filter * n_time_last_stage, 1),
        )

        blocks.append(CriticBlock(final_conv, critic_in))

        return blocks
