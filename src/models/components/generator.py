# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
from torch import Tensor
import numpy as np
from typing import List

from src.models.components.modules import PixelNorm, ConvBlockBN, PrintLayer, WS

class GeneratorBlock(nn.Module):
    """
    Description
    ----------
    Single stage for the progressive growing generator.
    Each stage consists of two possible actions:
    
    convolution_sequence: data always runs through this block 
    
    out_sequence: if the current stage is the last stage, data 
    gets passed through here. This is needed since the number of
    filter might not match the number of channels in the data

    Parameters:
    ----------
    intermediate_sequence : nn.Sequence
        Sequence of modules that process stage

    out_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current output

    resample_sequence  : nn.Sequence
        Sequence of modules between stages to upsample data
    """

    def __init__(self, intermediate_sequence:nn.Sequential, out_sequence:nn.Sequential) -> None:
        super(GeneratorBlock, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.out_sequence = out_sequence

    def forward(self, x, last=False, **kwargs):
        out = self.intermediate_sequence(x, **kwargs)
        if last:
            out = self.out_sequence(out, **kwargs)
        return out
    
    def stage_requires_grad(self, requires_grad:bool):
        for module in [self.intermediate_sequence, self.out_sequence]:
            for param in module.parameters():
                param.requires_grad = requires_grad

class Generator(nn.Module):
    """
    Description
    ----------
    Abstract base class used to build new Generators for implementing GANs.

    Parameters:
    ----------
    n_classes : int
        Number of classes
    latent_dim : int
        Dimension of the latent vector
    embedding_dim : int
        Dimension of the label embedding
    fading : bool
        If fading is used
    freeze : bool
        If parameters of past stages are frozen
    """

    def __init__(self,
                 n_classes:int,
                 latent_dim:int,
                 embedding_dim:int,
                 fading:bool=False,
                 freeze:bool=False,
                 **kwargs
                 ) -> None:
        
        super().__init__(**kwargs)
        
        self.blocks:List[GeneratorBlock] = list()

        self.label_embedding = nn.Embedding(n_classes, embedding_dim)
        self.embedding_dim = embedding_dim
        self.fading = fading
        self.freeze = freeze
        self.latent_dim = latent_dim  
        self.n_classes = n_classes

    def set_stage(self, cur_stage:int):
        self.cur_stage = cur_stage
        self._stage = self.cur_stage - 1 # Internal stage variable. Differs from GAN stage (cur_stage).

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
        
        # In the first stage we do not need fading and therefore set alpha to 1
        if self.cur_stage == 1:
            self.alpha = 1
        else:
            self.alpha = 0


    def forward(self, X, y, **kwargs):
        embedding = self.label_embedding(y)
        #embedding shape: batch_size x 10
        X = torch.cat([X, embedding], dim=1)

        for i in range(0, self.cur_stage):
            last = (i == self._stage)
            if last and self.fading and self.alpha < 1:
                # if this is the last stage, fading is active and alpha < 1
                # we copy the output of the previous stage, upsample it
                # and interpolate it with the output of the current stage.
                X_ = self.blocks[i-1].out_sequence(X, **kwargs)
                X_ = self.resample(X_, X.shape[-1]*2)

                # pass X through last stage
                X = self.blocks[i](X, last=last, **kwargs)

                # interpolate
                X = self.alpha * X + (1 - self.alpha) * X_
            else:
                X = self.blocks[i](X, last=last, **kwargs)
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

    def generate(self, shapes):

        z = torch.randn(shapes[0], self.latent_dim)

        y_fake = torch.randint(low=0, high=self.n_classes,
                               size=(shapes[0],), dtype=torch.int32)
        
        z, y_fake = self._to_current_device([z, y_fake])
        
        X_fake = self.forward(z, y_fake)

        return X_fake, y_fake
    
    def _to_current_device(self, inputs:List[torch.Tensor]):
        device = self.parameters().__next__().device
        return [x.to(device) for x in inputs]
   
    

    def build(self) -> List[GeneratorBlock]:
        r"""
        This function builds the generator blocks for the progressive growing GAN.
        """
        raise NotImplementedError("This function has to be implemented by the subclass")

    def description(self) -> None:
        r"""
        When implementing a new generator, this function can be overwritten to provide a description of the model.
        """
        print(
        r"""
        No Description Set    
        """
        )
