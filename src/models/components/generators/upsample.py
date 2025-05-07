# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import math
from typing import List

import numpy as np
import torch.nn as nn

from src.models.components.generator import Generator, GeneratorBlock
from src.models.components.modules import WS, PixelNorm, create_multiconv_for_stage


class Generator(Generator):
    def __init__(
        self,
        n_filter: int,
        n_samples: int,
        n_stages: int,
        n_channels: int,
        current_stage: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.blocks: List[GeneratorBlock] = self.build(
            n_filter, n_samples, n_stages, n_channels, **kwargs
        )

        # set stage
        self.set_stage(current_stage)

        # Dont train the linear layer in the generator:
        # self.blocks[0].intermediate_sequence[0].requires_grad_(False)

    def build(
        self,
        n_filter,
        n_samples,
        n_stages,
        n_channels,
        latent_dim,
        embedding_dim,
        kernel_size=3,
        **kwargs,
    ) -> List[GeneratorBlock]:
        # Generator:
        n_time_first_layer = int(np.floor(n_samples / 2 ** (n_stages - 1)))
        blocks = nn.ModuleList()

        upsample = nn.Sequential(
            WS(nn.ConvTranspose1d(n_filter, n_filter, 4, stride=2, padding=1)),  # WS()
            nn.LeakyReLU(0.2),
        )

        generator_out = WS(
            nn.Conv1d(
                n_filter,
                n_channels,
                kernel_size=1,
            )
        )  # WS()

        n_time_first_layer_downsampled = int(np.floor(n_samples / 2 ** (n_stages + 1)))

        head_layers = [
            WS(
                nn.Linear(
                    latent_dim + embedding_dim,
                    n_filter * n_time_first_layer_downsampled,
                )
            ),
            nn.Unflatten(1, (n_filter, n_time_first_layer_downsampled)),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        ]

        # Part 2: Convolutional Upsampling
        upsampling_layers_list = []
        current_time_steps = n_time_first_layer_downsampled
        num_upsample_blocks = int(
            math.log2(n_time_first_layer / n_time_first_layer_downsampled)
        )

        if (
            n_time_first_layer / n_time_first_layer_downsampled
            != 2**num_upsample_blocks
        ):
            raise ValueError(
                "n_time_first_layer / initial_time_steps must be a power of 2 for this simple stride=2 upsampling."
            )

        for i in range(num_upsample_blocks):
            # Each block doubles the time steps
            upsampling_layers_list.extend(
                [
                    WS(
                        nn.ConvTranspose1d(
                            n_filter, n_filter, kernel_size=4, stride=2, padding=1
                        )
                    ),
                    nn.LeakyReLU(0.2),
                    PixelNorm(),
                ]
            )
            current_time_steps *= 2

        # The rest of the original first_conv block
        tail_layers = [
            create_multiconv_for_stage(n_filter, 1),  # Assuming stage_idx=1 or similar
            nn.LeakyReLU(0.2),
            PixelNorm(),
            WS(nn.Conv1d(n_filter, n_filter, kernel_size=1, padding=0)),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        ]

        first_conv = nn.Sequential(
            *(head_layers + upsampling_layers_list + tail_layers)
        )

        if "batch_norm" in kwargs and kwargs["batch_norm"] == True:
            first_conv.add_module(nn.BatchNorm1d(n_filter))

        blocks.append(GeneratorBlock(first_conv, generator_out))

        for stage in range(2, n_stages + 1):
            stage_conv = nn.Sequential(
                *upsample,
                create_multiconv_for_stage(n_filter, stage),
                nn.LeakyReLU(0.2),
                PixelNorm(),
                WS(nn.Conv1d(n_filter, n_filter, kernel_size=1, padding=0)),
                nn.LeakyReLU(0.2),
                PixelNorm(),
            )

            # Out sequence is independent of stage
            blocks.append(GeneratorBlock(stage_conv, generator_out))

        return nn.ModuleList(blocks)

    def description(self) -> None:
        print(
            r"""
            Baseline generator, this one follows the architecture of the master thesis of Samuel Boehm
            and therefor should show similar results!
            """
        )
