# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import  numpy as np
from gan.model.modules import ConvBlockStage, PrintLayer, WS


class CriticStage(nn.Module):
    """
    Description
    ----------
    Single stage for the progressive growing critic.
    Each stage consists of two possible actions:
    
    convolution_sequence: data always runs through this block 
    
    in_sequence: if the current stage is the first stage, data 
    gets passed through here. This is needed since the number of
    filters might not match the number of channels in the data
    and therefore a convolution is used to match the number of
    filters. 

    Attributes
    ----------
    convolution_sequence : nn.Sequential
        Sequence of modules that process stage

    in_sequence : nn.Sequential
        Sequence of modules that is applied if stage is the current input
    
    resample_sequence : nn.Sequential
        sequence of modules between stages to downsample data
    """

    def __init__(self, intermediate_sequence:nn.Sequential, in_sequence:nn.Sequential, resample_sequence:nn.Sequential):
        super(CriticStage, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = in_sequence
        self.resample = resample_sequence
    

    def forward(self, x, first=False, **kwargs):
        if first:
            x = self.in_sequence(x, **kwargs)
        out = self.intermediate_sequence(x, **kwargs)
        return out
    
    def stage_requires_grad(self, requires_grad:bool):
        for module in [self.intermediate_sequence, self.in_sequence, self.resample]:
            for param in module.parameters():
                param.requires_grad = requires_grad
    
    def __repr__(self):
        return f'CriticStage kernel size: {self.intermediate_sequence[0].kernel_size}'


class Critic(nn.Module):
    """
    Critic module for implementing progressive GANs

    Attributes
    ----------
    blocks : list
        List of `CriticStage`s. Each represent one
        stage during progression
    
    Parameters
    ----------
    blocks : list
        List of `ProgressiveDiscriminatorBlock` which each represent one
        stage during progression
    """

    def __init__(self, n_time, n_channels, n_classes, blocks, stage=1, fading=False, freeze=False):
        super(Critic, self).__init__()
        self.blocks  = nn.ModuleList(blocks)
        self.n_time = n_time
        self.fading = fading
        self.freeze = freeze
        self.label_embedding = nn.Embedding(n_classes, n_time)
        self.alpha = 0
        self.set_stage(stage)


    def set_stage(self, stage):
        # Each time a new stage is set, some parameters need to be updated:
        ## 1. The current stage:
        self.cur_stage = stage

        # Internal stage variable. Differs from GAN stage (cur_stage).
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

        ## 3. Special case for fading in the first stage:
        # In the first stage we do not need fading and therefore set alpha to 1
        if self.cur_stage == 1:
            self.alpha = 1
        else:
            self.alpha = 0

    def forward(self, x:torch.Tensor, y:torch.Tensor, **kwargs):
        
        embedding:torch.Tensor = self.label_embedding(y).view(y.shape[0], 1, self.n_time)
        embedding = self.downsample(embedding, self._stage)
        
        x = torch.cat([x, embedding], 1) # batch_size x n_channels + 1 x n_time 

        for i in range(self._stage, len(self.blocks)):
            first = (i == self._stage)
            if first and self.fading and self.alpha < 1:
                # if this is the first stage, fading is used and alpha < 1
                # we take the current input, downsample it for the next block
                # match dimensions by using in_sequence and interpolate with
                # the current bock output with the downsampled input. 
                x_ = self.downsample(x, 1)
                x_ = self.blocks[i-1].in_sequence(x_, **kwargs)

                # pass x through new (current) block
                x = self.blocks[i](x, first=first,  **kwargs)
                # interpolate x_ and x
                x = self.alpha * x + (1 - self.alpha) * x_  
            else:
                x = self.blocks[i](x,  first=first, **kwargs)
        return x
    
    def downsample(self, x, steps):
        """
        Downscale input. Using bicubic interpolation.
    
        Parameters
        ----------
        x : tensor
            Input data
        steps : int
            for each step we downsample by a factor of 0.5

        Returns
        -------
        x : tensor
            Downsampled data
        """

        x = torch.unsqueeze(x, 0)
        for i in range(steps):
           x = nn.functional.interpolate(x, scale_factor=(1, 0.5), mode='bicubic')
        x = torch.squeeze(x, 0)
        return x
    

def build_critic(n_filters, n_time, n_stages, n_channels, n_classes, fading, freeze):

    n_channels += 1 # Add one channel for embedding

    # Calculate the number of timepoints in the last layer
    # n_stages - 1 since we dont downsample after the last convolution
    n_time_last_stage = int(np.floor(n_time / 2 ** (n_stages - 1)))
    
    # Critic:
    blocks = nn.ModuleList()

        
    critic_in = nn.Sequential(
        WS(nn.Conv1d(n_channels, n_filters, kernel_size=1, stride=1)), #WS()
        nn.LeakyReLU(0.2),
    )

    downsample = nn.Sequential(nn.ReflectionPad1d(1),
                                WS(nn.Conv1d(n_filters, n_filters, kernel_size=4, stride=2)), #WS()
                                nn.LeakyReLU(0.2))

    for stage in range(1, n_stages):
        stage_conv = nn.Sequential(
                    ConvBlockStage(n_filters, n_stages - stage, generator=False),
                    downsample)

        # In sequence is independent of stage
        blocks.append(CriticStage(stage_conv, critic_in, downsample))


    
    last_conv = nn.Sequential(
        ConvBlockStage(n_filters, 0, generator=False),
        nn.Flatten(),
        nn.Linear(n_filters * n_time_last_stage, 1),
    )

    blocks.append(CriticStage(last_conv, critic_in, downsample))


    
    return Critic(n_time, n_channels, n_classes, blocks, fading=fading, freeze=freeze)