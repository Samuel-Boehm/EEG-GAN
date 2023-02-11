# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import List

from torch import nn
import torch

import sys

from eeggan.pytorch.modules.module import Module
from eeggan.training.generator import Generator

"""
Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
Progressive Growing of GANs for Improved Quality, Stability, and Variation.
Retrieved from http://arxiv.org/abs/1710.10196
"""


class ProgressiveGeneratorBlock(Module):
    """
    Block for one Generator stage during progression

    Attributes
    ----------
    intermediate_sequence : nn.Sequence
        Sequence of modules that process stage
    out_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current output
    fade_sequence : nn.Sequence
        Sequence of modules that is used for fading stage into output
    """

    def __init__(self, intermediate_sequence, out_sequence, fade_sequence):
        super(ProgressiveGeneratorBlock, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.out_sequence = out_sequence
        self.fade_sequence = fade_sequence

    def forward(self, x, last=False):
        out = self.intermediate_sequence(x)
        if last:
            out = self.out_sequence(out)
        return out


class ProgressiveConditionalGenerator(Generator):
    """
    Generator module for implementing progressive GANS

    Attributes
    ----------
    blocks : list
        List of `ProgressiveGeneratorBlock` which each represent one
        stage during progression
    cur_block : int
        Current stage of progression (from first to last)
    alpha : float
        Fading parameter. Defines how much of the second to last stage gets
        merged into the output.

    Parameters
    ----------
    blocks : list
        List of `ProgressiveGeneratorBlock` which each represent one
        stage during progression
    """

    def __init__(self, n_samples, n_channels, n_classes, n_latent, blocks: List[ProgressiveGeneratorBlock], n_subjects):
        super(ProgressiveConditionalGenerator, self).__init__(n_samples, n_channels, n_classes, n_latent)
        # noinspection PyTypeChecker
        self.blocks: List[ProgressiveGeneratorBlock] = nn.ModuleList(blocks)
        self.cur_block = 0
        self.alpha = 1.

        self.label_embedding = nn.Embedding(n_classes, 25)
        self.subject_embedding = nn.Embedding(n_subjects, 75)

    def forward(self, x, y):
    
        label = y[:,0].type(torch.int)
        subject = y[:,1].type(torch.int)
        

        label_emb = self.label_embedding(label)
        subject_emb = self.subject_embedding(subject)
        #embedding shape: batch_size x 25

        x = torch.cat([x, label_emb, subject_emb], dim=1)
        # input shape: batch_size x (latent_size + 25 + 75)

        fade = False
        alpha = self.alpha
        for i in range(0, self.cur_block + 1):
            x = self.blocks[i](x, last=(i == self.cur_block))
            if alpha < 1. and i == self.cur_block - 1:
                tmp = self.blocks[i].out_sequence(x)
                fade = True

        if fade:
            tmp = self.blocks[i - 1].fade_sequence(tmp)
            x = alpha * x + (1. - alpha) * tmp

        return x

    def upsample_from_block(self, x, i_block):
        """
        Scales up input to the size of current input stage.
        Utilizes `ProgressiveGeneratorBlock.fade_sequence` from each stage.

        Parameters
        ----------
        x : autograd.Variable
            Input data
        i_block : int
            Stage to which input should be upwnsampled

        Returns
        -------
        output : autograd.Variable
            Upsampled data
        """
        for i in range(i_block, len(self.blocks) - 1):
            x = self.blocks[i].fade_sequence(x)
        output = x
        return output
