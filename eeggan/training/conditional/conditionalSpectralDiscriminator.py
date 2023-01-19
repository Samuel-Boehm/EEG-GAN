# coding=utf-8

#  Author: Samuel Böhm <samuel-boehm@web.de>

from typing import List
import sys

from torch import nn
import torch

from eeggan.training.discriminator import Discriminator
from eeggan.training.conditional.conditionalDiscriminator import ProgressiveDiscriminatorBlock

"""
Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
Progressive Growing of GANs for Improved Quality, Stability, and Variation.
Retrieved from http://arxiv.org/abs/1710.10196
"""

class ProgressiveSpectralDiscriminator(Discriminator):
    """
    Discriminator module for implementing progressive GANS

    Attributes
    ----------
    blocks : list
        List of `ProgressiveDiscriminatorBlock` which each represent one
        stage during progression
    cur_block : int
        Current stage of progression (from last to first)
    alpha : float
        Fading parameter. Defines how much of the input skips the current block

    Parameters
    ----------
    blocks : list
        List of `ProgressiveDiscriminatorBlock` which each represent one
        stage during progression
    """

    def __init__(self, n_samples, n_channels, n_classes, blocks: List[ProgressiveDiscriminatorBlock]):
        super(ProgressiveSpectralDiscriminator, self).__init__(n_samples, n_channels, n_classes)
        self.blocks: List[ProgressiveDiscriminatorBlock] = nn.ModuleList(blocks)
        self.cur_block = len(self.blocks) - 1
        self.alpha = 1.
        self.n_samples = n_samples

        self.label_embedding = nn.Embedding(n_classes, self.n_samples)



    def forward(self, x, y, **kwargs):

        fade = False
        alpha = self.alpha
        y = y.type(torch.int)
        
        embedding = self.label_embedding(y).view(y.shape[0], 1, self.n_samples)
        embedding = self.downsample_to_block(embedding, self.cur_block)
        # embedding = embedding[:, : , :x.shape[-1]]

        x = torch.cat([x, embedding], 1) #batch_size x n_channels + 1 x n_samples 
        
        x = self.fft(x)

        for i in range(self.cur_block, len(self.blocks)):
            if alpha < 1. and i == self.cur_block:
                tmp = self.blocks[i].fade_sequence(x, **kwargs)
                tmp = self.blocks[i + 1].in_sequence(tmp, **kwargs)
                fade = True

            if fade and i == self.cur_block + 1:
                x = alpha * x + (1. - alpha) * tmp
            x = self.blocks[i](x,  first=(i == self.cur_block), **kwargs)
        return x

    def fft(self, data,  **kwargs):
        fft_out = torch.fft.rfft(data)
        fft_out = torch.abs(fft_out)
        fft_out = torch.log(fft_out)
        # average over batch (size: ch x input_length)
        # fft_out = torch.mean(fft_out, (0))
        # normalize between 0, 1
        # fft_out = fft_out - fft_out.min(2)[0].view(-1,1)
        # fft_out = fft_out / fft_out.max(2)[0].view(-1,1)

        return fft_out

    def downsampdownsample_to_blockle_embedding(self, x, i_block):
        """
        Scales down input to the size of current input stage.
        Utilizes `ProgressiveDiscriminatorBlock.fade_sequence` from each stage.

        Parameters
        ----------
        x : autograd.Variable
            Input data
        i_block : int
            Stage to which input should be downsampled

        Returns
        -------
        output : autograd.Variable
            Downsampled data
        """
        for i in range(i_block):
            x = self.blocks[i].fade_sequence(x)
        output = x
        return output
