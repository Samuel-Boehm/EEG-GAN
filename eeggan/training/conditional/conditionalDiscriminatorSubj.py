# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import List
import sys

from torch import nn
import torch

from eeggan.pytorch.modules.module import Module
from eeggan.training.discriminator import Discriminator

"""
Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
Progressive Growing of GANs for Improved Quality, Stability, and Variation.
Retrieved from http://arxiv.org/abs/1710.10196
"""


class ProgressiveDiscriminatorBlock(Module):
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

    def forward(self, x, first=False, **kwargs):
        if first:
            x = self.in_sequence(x, **kwargs)
        out = self.intermediate_sequence(x, **kwargs)
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

    def __init__(self, n_samples, n_channels, n_classes, blocks: List[ProgressiveDiscriminatorBlock],n_subjects):
        super(ProgressiveConditionalDiscriminator, self).__init__(n_samples, n_channels, n_classes)
        # noinspection PyTypeChecker
        self.blocks: List[ProgressiveDiscriminatorBlock] = nn.ModuleList(blocks)
        self.cur_block = len(self.blocks) - 1
        self.alpha = 1.
        self.n_samples = n_samples

        self.label_embedding = nn.Embedding(n_classes, self.n_samples)
        self.subject_embedding = nn.Embedding(n_subjects, self.n_samples)



    def forward(self, x, y, **kwargs):

        fade = False
        alpha = self.alpha

        label = y[:,0].type(torch.int)
        subject = y[:,1].type(torch.int)
        
        label_emb = self.label_embedding(label).view(y.shape[0], 1, self.n_samples)
        subject_emb = self.label_embedding(subject).view(y.shape[0], 1, self.n_samples)

        embedding = torch.cat([label_emb, subject_emb], 1)

        embedding = self.downsample_to_block(embedding, self.cur_block)

        x = torch.cat([x, embedding], 1) #batch_size x n_channels + 2 x n_samples 

        for i in range(self.cur_block, len(self.blocks)):
            if alpha < 1. and i == self.cur_block:
                tmp = self.blocks[i].fade_sequence(x, **kwargs)
                tmp = self.blocks[i + 1].in_sequence(tmp, **kwargs)
                fade = True

            if fade and i == self.cur_block + 1:
                x = alpha * x + (1. - alpha) * tmp
            x = self.blocks[i](x,  first=(i == self.cur_block), **kwargs)
  
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
