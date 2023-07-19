# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import numpy as np

from GAN.Model.Modules import PixelNorm, ConvBlockStage, PrintLayer, WS

class GeneratorStage(nn.Module):
    """
    Description
    ----------
    Single stage for the progressive growing generator.
    Each stage consists of two possible actions:
    
    convolution_sequence: data always runs through this block 
    
    out_sequence: if the current stage is the last stage, data 
    gets passed through here. This is needed since the number of
    filter might not match the number of channels in the data



    Attributes
    ----------
    intermediate_sequence : nn.Sequence
        Sequence of modules that process stage

    out_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current output

    resample_sequence  : nn.Sequence
        Sequence of modules between stages to upsample data
    """

    def __init__(self, intermediate_sequence, out_sequence, resample_sequence):
        super(GeneratorStage, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.out_sequence = out_sequence
        self.resample = resample_sequence

    def forward(self, x, last=False, **kwargs):
        out = self.intermediate_sequence(x, **kwargs)
        if last:
            out = self.out_sequence(out, **kwargs)
        return out


class Generator(nn.Module):
    """
    Description
    ----------
    Generator module for implementing progressive GANs

    Attributes
    ----------
    blocks : list
        List of `GeneratorStage`s. Each represent one
        stage during progression
    n_classes: int
        number of classes, required to create embedding layer 
        for conditional GAN
    """

    def __init__(self, blocks, n_classes, embedding_dim, stage=1, fading=False):
        super(Generator, self).__init__()
        self.blocks = nn.ModuleList(blocks)
        # set stage
        self.set_stage(stage)
        self.label_embedding = nn.Embedding(n_classes, embedding_dim)
        self.fading = fading

    def set_stage(self, stage):
        self.cur_stage = stage
        self._stage = self.cur_stage - 1 # Internal stage variable. Differs from GAN stage (cur_stage).
        
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
    
def build_generator(n_filters, n_time, n_stages, n_channels, n_classes,
                    latent_dim, embedding_dim, fading) -> Generator:
    
    
    # Generator:
    n_time_first_layer = int(np.floor(n_time / 2 ** (n_stages-1)))
    blocks = []

    # Note that the first conv stage in the generator differs from the others
    # because it takes the latent vector as input
    first_conv = nn.Sequential(
        nn.Linear(latent_dim + embedding_dim, n_filters * n_time_first_layer),
        nn.Unflatten(1, (n_filters, n_time_first_layer)),
        nn.LeakyReLU(0.2),
        PixelNorm(),
        ConvBlockStage(n_filters, 0, generator=True),
        )
    
    upsample = nn.Sequential(
                WS(nn.ConvTranspose1d(n_filters, n_filters, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2)
        )
    
    generator_out = WS(nn.Conv1d(n_filters, n_channels, kernel_size=1,))




    blocks.append(GeneratorStage(first_conv, generator_out, upsample))

    for stage in range(1, n_stages):
        stage_conv = nn.Sequential(
            upsample,
            ConvBlockStage(n_filters, stage, generator=True),
        )

        # Out sequence is independent of stage
        blocks.append(GeneratorStage(stage_conv, generator_out, upsample))
        
    return Generator(blocks, n_classes, embedding_dim, fading=fading)


