# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import numpy as np
from typing import List

from src.models.components.modules import PixelNorm, ConvBlock, PrintLayer, WS
from src.models.components.generator import GeneratorBlock, Generator


class Generator(Generator):
    def __init__(self,
                 n_filter:int,
                 n_samples:int,
                 n_stages:int,
                 n_channels:int,
                 latent_dim:int,
                 embedding_dim:int,
                 current_stage:int=1,
                 **kwargs
                 ) -> None:
        
        super(Generator, self).__init__()
        
        self.blocks:List[GeneratorBlock] = self.build(
            n_filter, n_samples, n_stages, n_channels, latent_dim, embedding_dim
            )

        # set stage
        self.set_stage(current_stage)

        # Dont train the linear layer in the generator: 
        self.blocks[0].intermediate_sequence[0].requires_grad_(False)


    def build(self, n_filter, n_samples, n_stages, n_channels,
                        latent_dim, embedding_dim, kernel_size=3) -> List[GeneratorBlock]:
        
        # Generator:
        n_time_first_layer = int(np.floor(n_samples / 2 ** (n_stages-1)))
        blocks = nn.ModuleList()

        # Note that the first conv stage in the generator differs from the others
        # because it takes the latent vector as input
        first_conv = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, n_filter * n_time_first_layer),
            nn.Unflatten(1, (n_filter, n_time_first_layer)),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            ConvBlock(n_filter, 0, kernel_size=kernel_size, is_generator=True),
            )
        
        upsample = nn.Sequential(
                    WS(nn.ConvTranspose1d(n_filter, n_filter, 4, stride=2, padding=1)), #WS()
                    nn.LeakyReLU(0.2)
            )
        
        generator_out = WS(nn.Conv1d(n_filter, n_channels, kernel_size=1,)) #WS()

        blocks.append(GeneratorBlock(first_conv, generator_out))

        for stage in range(1, n_stages):
            _kernel_size = int(kernel_size + (stage*4))
            stage_conv = nn.Sequential(
                upsample,
                ConvBlock(n_filter, stage, kernel_size=_kernel_size, is_generator=True),
            )

            # Out sequence is independent of stage
            blocks.append(GeneratorBlock(stage_conv, generator_out))
            
        return nn.ModuleList(blocks) 