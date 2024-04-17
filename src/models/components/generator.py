# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import numpy as np
from typing import List

from gan.model.modules import PixelNorm, ConvBlock, PrintLayer, WS




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
    Generator module for implementing progressive GANs

    Parameters:
    ----------
    n_filter : int
        Number of filters in the convolutional layers
    n_time : int
        Number of timepoints in the input data
    n_stages : int
        Number of stages in the generator
    n_channels : int
        Number of channels in the input data
    n_classes : int
        Number of classes
    latent_dim : int
        Dimension of the latent vector
    embedding_dim : int
        Dimension of the label embedding
    cur_stage : int
        Current stage of the generator
    fading : bool
        If fading is used
    freeze : bool
        If parameters of past stages are frozen
    """

    def __init__(self,
                 n_filter:int,
                 n_time:int,
                 n_stages:int,
                 n_channels:int,
                 n_classes:int,
                 latent_dim:int,
                 embedding_dim:int,
                 cur_stage:int=1,
                 fading:bool=False,
                 freeze:bool=False
                 ) -> None:
        
        super(Generator, self).__init__()
        
        self.blocks:List[GeneratorBlock] = self.build(
            n_filter, n_time, n_stages, n_channels, latent_dim, embedding_dim
            )
        self.label_embedding = nn.Embedding(n_classes, embedding_dim)
        self.fading = fading
        self.freeze = freeze
        self.latent_dim = latent_dim  
        self.n_classes = n_classes

        # set stage
        self.set_stage(cur_stage)


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


    def forward(self, x, y, **kwargs):
        embedding = self.label_embedding(y)
        #embedding shape: batch_size x 10
        x = torch.cat([x, embedding], dim=1)

        for i in range(0, self.cur_stage):
            last = (i == self._stage)
            if last and self.fading and self.alpha < 1:
                # if this is the last stage, fading is active and alpha < 1
                # we copy the output of the previous stage, and upsample it
                # and interpolate it with the output of the current stage.
                x_ = self.blocks[i-1].out_sequence(x, **kwargs)
                x_ = self.upsample(x_, 1)

                # pass x through last stage
                x = self.blocks[i](x, last=last, **kwargs)

                # interpolate
                x = self.alpha * x + (1 - self.alpha) * x_
            else:
                x = self.blocks[i](x, last=last, **kwargs)
        return x


    def upsample(self, x, steps):
        """
        Upsample input.
        uses bicubic interpolation. 

        Parameters
        ----------
        x : tensor
            Input data
        steps : int
            for each step, the data is upsampled by a factor of 2

        Returns
        -------
        output : tensor
            Upsampled data
        """
        x = torch.unsqueeze(x, 0)
        for i in range(steps):
           x = nn.functional.interpolate(x, scale_factor=(1, 2), mode='bicubic')
        x = torch.squeeze(x, 0)
        return x
    
    def build(self, n_filter, n_time, n_stages, n_channels,
                        latent_dim, embedding_dim) -> List[GeneratorBlock]:
        
        
        # Generator:
        n_time_first_layer = int(np.floor(n_time / 2 ** (n_stages-1)))
        blocks = nn.ModuleList()

        # Note that the first conv stage in the generator differs from the others
        # because it takes the latent vector as input
        first_conv = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, n_filter * n_time_first_layer),
            nn.Unflatten(1, (n_filter, n_time_first_layer)),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            ConvBlock(n_filter, 0, is_generator=True),
            )
        
        upsample = nn.Sequential(
                    WS(nn.ConvTranspose1d(n_filter, n_filter, 4, stride=2, padding=1)), #WS()
                    nn.LeakyReLU(0.2)
            )
        
        generator_out = WS(nn.Conv1d(n_filter, n_channels, kernel_size=1,)) #WS()

        blocks.append(GeneratorBlock(first_conv, generator_out))

        for stage in range(1, n_stages):
            stage_conv = nn.Sequential(
                upsample,
                ConvBlock(n_filter, stage, is_generator=True),
            )

            # Out sequence is independent of stage
            blocks.append(GeneratorBlock(stage_conv, generator_out))
            
        return nn.ModuleList(blocks) 
    
    def generate(self, shapes):

        z = torch.randn(shapes[0], self.latent_dim)
        # z = z.type_as(X_real)
        y_fake = torch.randint(low=0, high=self.n_classes,
                               size=(shapes[0],), dtype=torch.int32)
        
        X_fake = self.forward(z, y_fake)

        return X_fake, y_fake

   