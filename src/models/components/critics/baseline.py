# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from typing import List

import numpy as np
import torch.nn as nn

from src.models.components.critic import Critic, CriticBlock
from src.models.components.modules import (
    WS,
    create_multiconv_for_stage,
)


class Critic(Critic):
    r"""
    Default critic for EEG-GAN
    """

    def __init__(
        self,
        n_filter: int,
        n_stages: int,
        n_channels: int,
        current_stage: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.blocks = self.build(
            n_filter=n_filter, n_stages=n_stages, n_channels=n_channels, **kwargs
        )
        self.set_stage(current_stage)

    def build(
        self,
        n_filter: int,
        n_samples: int,
        n_stages: int,
        n_channels: int,
        kernel_size=3,
        **kwargs,
    ) -> List[CriticBlock]:
        n_channels += 1  # Add one channel for embedding

        # Calculate the number of timepoints in the last layer
        # n_stages - 1 since we dont downsample after the last convolution
        n_time_last_stage = int(np.floor(n_samples / 2 ** (n_stages - 1)))

        # Critic:
        blocks = nn.ModuleList()

        critic_in = nn.Sequential(
            WS(nn.Conv1d(n_channels, n_filter, kernel_size=1, stride=1)),  # WS()
            nn.LeakyReLU(0.2),
        )

        downsample = nn.Sequential(
            nn.ReflectionPad1d(1),
            WS(nn.Conv1d(n_filter, n_filter, kernel_size=4, stride=2)),  # WS()
            nn.LeakyReLU(0.2),
        )

        for stage in range(n_stages, 1, -1):
            stage_conv = nn.Sequential(
                create_multiconv_for_stage(n_filter, stage),
                nn.LeakyReLU(0.2),
                WS(nn.Conv1d(n_filter, n_filter, kernel_size=1, padding=0)),
                nn.LeakyReLU(0.2),
                *downsample,
            )

            # In sequence is independent of stage
            blocks.append(CriticBlock(stage_conv, critic_in))

        final_conv = nn.Sequential(
            create_multiconv_for_stage(n_filter, 1),
            nn.LeakyReLU(0.2),
            WS(nn.Conv1d(n_filter, n_filter, kernel_size=1, padding=0)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(n_filter * n_time_last_stage, 1),
        )

        blocks.append(CriticBlock(final_conv, critic_in))

        return blocks

    def description(self) -> None:
        print(
            r"""
            Baseline critic, this one follows the architecture of the master thesis of Samuel Boehm
            and therefor should show similar results!
            """
        )
