# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import  numpy as np
from Model.Modules import ConvBlockStage, PrintLayer, WS


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

    def __init__(self, intermediate_sequence:nn.Sequential, out_sequence:nn.Sequential, resample_sequence:nn.Sequential):
        super(CriticStage, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = out_sequence
        self.resample = resample_sequence
    

    def forward(self, x, first=False, alpha=1, fading=False, **kwargs):
        
        if first:
            x = self.in_sequence(x, **kwargs)
            if fading and alpha < 1:
                # downsample data directly after in_sequence
                ds_x = self.resample(x, **kwargs)
                # pass data through intermediate_sequence
                fx = self.intermediate_sequence(x, **kwargs)
                # interpolate between x and fx
                out = (1-alpha)*ds_x + alpha*fx
                # return interpolated data
                return out
        # if alpha >=1 or first == False:
        out = self.intermediate_sequence(x, **kwargs)
        return out


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

    def __init__(self, n_time, n_channels, n_classes, blocks, stage=1, fading=False):
        super(Critic, self).__init__()
        # noinspection PyTypeChecker
        self.blocks  = nn.ModuleList(blocks)
        self.set_stage(stage)
        self.n_time = n_time
        self.fading = fading

        self.label_embedding = nn.Embedding(n_classes, n_time)

        self.alpha = 0

    def set_stage(self, stage):
        self.cur_stage = stage
        self._stage = len(self.blocks) - self.cur_stage # Internal stage variable. Differs from GAN stage (cur_stage).

        # In the first stage we do not need fading and therefore set alpha to 1
        if self.cur_stage == 1:
            self.alpha = 1
        else:
            self.alpha = 0




    def forward(self, x:torch.Tensor, y:torch.Tensor, **kwargs):
        
        embedding:torch.Tensor = self.label_embedding(y).view(y.shape[0], 1, self.n_time)
        embedding = self.downsample_to_stage(embedding, self._stage)

        x = torch.cat([x, embedding], 1) # batch_size x n_channels + 1 x n_time 

        for i in range(self._stage, len(self.blocks)):
            x = self.blocks[i](x,  first=(i == self._stage), alpha=self.alpha, **kwargs)
        return x
    
    def downsample_to_stage(self, x, stage):
        """
        Scales down input to the size of current input stage.
    
        Parameters
        ----------
        x : tensor
            Input data
        stage : int
            Stage to which input should be downsampled

        Returns
        -------
        x : tensor
            Downsampled data
        """

        x = torch.unsqueeze(x, 0)
        for i in range(stage):
           x = nn.functional.interpolate(x, scale_factor=(1, 0.5), mode='bicubic')
        x = torch.squeeze(x, 0)
        return x
    

def build_critic(n_filters, n_time, n_stages, n_channels, n_classes):

    n_channels += 1 # Add one channel for embedding

    # Calculate the number of timepoints in the last layer
    # n_stages - 1 since we dont downsample after the last convolution
    n_time_last_stage = int(np.floor(n_time / 2 ** (n_stages - 1)))
    
    # Critic:
    blocks = nn.ModuleList()

        
    critic_in = nn.Sequential(
        WS(nn.Conv1d(n_channels, n_filters, kernel_size=1, stride=1)),
        nn.LeakyReLU(0.2),
    )

    downsample = nn.Sequential(nn.ReflectionPad1d(1),
                                WS(nn.Conv1d(n_filters, n_filters, kernel_size=4, stride=2)), 
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


    
    return Critic(n_time, n_channels, n_classes, blocks)