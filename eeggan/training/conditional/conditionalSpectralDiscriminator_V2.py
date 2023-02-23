# coding=utf-8

#  Author: Samuel Böhm <samuel-boehm@web.de>

from typing import List
import sys

import numpy as np
import torch
from torch import nn
from torch import Tensor

from eeggan.training.discriminator import Discriminator
from eeggan.training.conditional.conditionalDiscriminator import ProgressiveDiscriminatorBlock



class ProgressiveSpectralDiscriminator(Discriminator):
    """
    Discriminator module for implementing progressive GANS

    Attributes
    ----------
    blocks : list
        List of `ProgressiveSpectralDiscriminatorBlock` which each represent one
        stage during progression
    cur_block : int
        Current stage of progression (from last to first)
    alpha : float
        Fading parameter. Defines how much of the input skips the current block

    Parameters
    ----------
    blocks : list
        List of `ProgressiveSpectralDiscriminatorBlock` which each represent one
        stage during progression
    """

    def __init__(self, n_samples, n_channels, n_classes, blocks: List[ProgressiveDiscriminatorBlock]):
        super(ProgressiveSpectralDiscriminator, self).__init__(n_samples, n_channels, n_classes)
        self.blocks: List[ProgressiveDiscriminatorBlock] = nn.ModuleList(blocks)
        self.cur_block = len(self.blocks) - 1
        self.alpha = 1.
        self.n_samples = n_samples
        self.label_embedding = nn.Embedding(n_classes, self.n_samples)



    def forward(self, x, y):
        x = self.spectral_vector(x)

        for i in range(self.cur_block, len(self.blocks)):
            x = self.blocks[i](x,  first=(i == self.cur_block))
        return x

    def get_input(self, x, y):
        '''Returns the input before passing it through forwad
        this is used to plot the output of self.spectral_vector'''
        x = self.spectral_vector(x)

        return x



    def spectral_vector(self, x):
        """Assumes first dimension to be batch size."""
        fft = torch.fft.rfft(x)
        fft_abs = torch.abs(fft)
        fft_abs = fft_abs + 1E-8
        fft_abs = torch.log(fft_abs)
        fft_mean = fft_abs.mean(axis=(0, 1)).squeeze()
        return fft_mean


