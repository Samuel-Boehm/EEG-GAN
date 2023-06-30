# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import numpy as np

from Model.Modules import PixelNorm, ConvBlockStage, PrintLayer, WS

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

    def forward(self, x, last=False, alpha=1, **kwargs):
        '''
        forward pass through generator stage. Depending on the stage,
        we either pass the data through the intermediate_sequence or
        '''
        fx = self.intermediate_sequence(x, **kwargs)
        if last:
            if alpha < 1:
                # If alpha < 1, we upsample the data...
                x = self.resample(x, **kwargs)
                # ...then we interpolate between x and fx...
                interpolation = (1-alpha)*x + alpha * fx
                # ...and pass it through the out_sequence:
                out = self.out_sequence(interpolation, **kwargs)
                return out
            else:
                out = self.out_sequence(fx, **kwargs)
                return out
            
        return fx


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

    def __init__(self, blocks, n_classes, embedding_dim, stage=1):
        super(Generator, self).__init__()
        self.blocks = nn.ModuleList(blocks)
        # set stage
        self.set_stage(stage)
        self.label_embedding = nn.Embedding(n_classes, embedding_dim)

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
            x = self.blocks[i](x, last=(i == self._stage), alpha=self.alpha, **kwargs)
        
        # increase alpha:
        return x


    def upsample_to_stage(self, x, stage):
        """
        Scales up input to the size of current input stage.
        Utilizes `ProgressiveGeneratorBlock.fade_sequence` from each stage.

        Parameters
        ----------
        x : tensor
            Input data
        stage : int
            Stage to which input should be upwnsampled

        Returns
        -------
        output : tensor
            Upsampled data
        """
        raise NotImplementedError
    
def build_generator(n_filters, n_time, n_stages, n_channels, n_classes,
                    latent_dim, embedding_dim, ) -> Generator:
    
    
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
        
    return Generator(blocks, n_classes, embedding_dim)


