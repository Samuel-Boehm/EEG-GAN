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

        self.calculate_size_for_block()


    def forward(self, x, y, **kwargs):
        #y = y.type(torch.int)
        #embedding = self.label_embedding(y).view(y.shape[0], 1, self.n_samples)
        #embedding = self.downsample_to_block(embedding, self.cur_block)
        # embedding = embedding[:, : , :x.shape[-1]]
        #x = torch.cat([x, embedding], 1) #batch_size x n_channels + 1 x n_samples 

        x = self.spectral_vector(x)

        for i in range(self.cur_block, len(self.blocks)):
            x = self.blocks[i](x,  first=(i == self.cur_block), **kwargs)
        return x

    def get_input(self, x, y, **kwargs):
        '''Returns the input before passing it through forwad
        this is handy to plot the output of self.spectral_vector'''
        #y = y.type(torch.int)
        #embedding = self.label_embedding(y).view(y.shape[0], 1, self.n_samples)
        #embedding = self.downsample_to_block(embedding, self.cur_block)
        # embedding = embedding[:, : , :x.shape[-1]]
        #x = torch.cat([x, embedding], 1) #batch_size x n_channels + 1 x n_samples 

        x = self.spectral_vector(x)

        return x



    def spectral_vector(self, x, **kwargs):
        """Assumes first dimension to be batch size."""

        fft = torch.view_as_real(torch.fft.rfft(x))
        # abs of complex
        fft_abs = torch.sum(fft**2,dim=3)
        fft_abs = fft_abs + 1E-8
        fft_abs = 20*torch.log(fft_abs)

        fft = fft_abs \
                .unsqueeze(1) \
                .expand(-1,self.vector_length,-1,-1) # repeat img for each radius

        # apply mask and compute profile vector
        profile = (fft * self.mask).sum((2,3))
        # normalize profile into [0,1]
        profile = profile * self.mask_n
        profile = profile - profile.min(1)[0].view(-1,1)
        profile = profile / profile.max(1)[0].view(-1,1)
        
        return profile


    def calculate_size_for_block(self):
        n_samples = np.floor(self.n_samples / 2 ** (self.cur_block))
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(self.n_channels / 2)
        # number of cols after onesided fft
        cols_onesided = int(n_samples / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices((self.n_channels  ,cols_onesided)) - np.array([[[shift_rows]],[[0]]])
        r = np.sqrt(r[0,:,:]**2+r[1,:,:]**2)
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r,axes=0)
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        # repeat slice for each radius
        r = torch.from_numpy(r).expand(
            r_max+1,-1,-1
            )
        radius_to_slice = torch.arange(r_max+1).view(-1,1,1)
        # generate mask for each radius
        
        mask = torch.where(
            r==radius_to_slice,
            torch.tensor(1,dtype=torch.float),
            torch.tensor(0,dtype=torch.float)
            )
        # how man entries for each radius?
        mask_n = torch.sum(mask,axis=(1,2))
        mask = mask.unsqueeze(0) # add batch dimension
        # normalization vector incl. batch dimension
        mask_n = (1/mask_n.to(torch.float)).unsqueeze(0)
        self.vector_length = (r_max+1)
        self.register_buffer('mask', torch.Tensor(mask))
        self.register_buffer('mask_n', torch.Tensor(mask_n))
        
