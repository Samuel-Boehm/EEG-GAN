# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import List
import sys

from torch import nn
import torch
from eeggan.training.discriminator import Discriminator

"""
Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
Progressive Growing of GANs for Improved Quality, Stability, and Variation.
Retrieved from http://arxiv.org/abs/1710.10196
"""


class ProgressiveDiscriminatorBlock(nn.Module):
    """
    Block for one Discriminator stage during progression

    Attributes
    ----------
    intermediate_sequence : nn.Sequence
        Sequence of modules that process stage
    in_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current input
    fade_sequence : nn.Sequence
        Sequence of modules that is used for fading input into stage
    """

    def __init__(self, intermediate_sequence, in_sequence, fade_sequence):
        super(ProgressiveDiscriminatorBlock, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = in_sequence
        self.fade_sequence = fade_sequence

    def forward(self, x, first=False):
        if first:
            x = self.in_sequence(x)
        out = self.intermediate_sequence(x)
        return out


class ProgressiveConditionalDiscriminator(Discriminator):
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
        super(ProgressiveConditionalDiscriminator, self).__init__(n_samples, n_channels, n_classes)
        # noinspection PyTypeChecker
        self.blocks: List[ProgressiveDiscriminatorBlock] = nn.ModuleList(blocks)
        self.cur_block = len(self.blocks) - 1
        self.alpha = 1.
        self.n_samples = n_samples

        self.label_embedding = nn.Embedding(n_classes, self.n_samples)



    def forward(self, x, y):

        fade = False
        alpha = self.alpha
        y = y.type(torch.int)
        
        embedding = self.label_embedding(y).view(y.shape[0], 1, self.n_samples)

        embedding = self.downsample_to_block(embedding, self.cur_block)

        x = torch.cat([x, embedding], 1) # batch_size x n_channels + 1 x n_samples 

        for i in range(self.cur_block, len(self.blocks)):
            if alpha < 1. and i == self.cur_block:
                tmp = self.blocks[i].fade_sequence(x)
                tmp = self.blocks[i + 1].in_sequence(tmp)
                fade = True

            if fade and i == self.cur_block + 1:
                x = alpha * x + (1. - alpha) * tmp
            x = self.blocks[i](x,  first=(i == self.cur_block))
  
        return x

    def downsample_to_block(self, x, i_block):
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
