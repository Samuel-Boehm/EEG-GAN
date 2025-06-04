# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch
import torch.nn as nn


class SpectralCriticBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, 32), nn.LeakyReLU(0.2), nn.Linear(32, out_size)
        )

    def forward(self, x):
        return self.fc(x)


class SpectralCritic(nn.Module):
    def __init__(
        self, n_samples: int, n_stages: int, current_stage=1, **kwargs
    ) -> None:
        super(SpectralCritic, self).__init__()
        self.blocks = self.build(n_samples, n_stages)
        self.set_stage(current_stage)

        self.alpha = 0

    def set_stage(self, stage):
        self.cur_stage = stage
        self._stage = (
            len(self.blocks) - self.cur_stage
        )  # Internal stage variable. Differs from GAN stage (cur_stage).
        # In the first stage we do not need fading and therefore set alpha to 1
        if self.cur_stage == 1:
            self.alpha = 1
        else:
            self.alpha = 0

    def forward(self, x, y):
        # x: (batch, channels, time)
        fft = torch.fft.rfft(x, dim=-1)
        fft_abs = torch.log(torch.abs(fft) + 1e-8)
        fft_feat = fft_abs.mean(dim=1)  # (batch, freq_bins)
        label_emb = self.label_embedding(y)  # (batch, emb_dim)
        fft_feat = torch.cat([fft_feat, label_emb], dim=1)
        out = self.blocks[self._stage](fft_feat)
        return out

    def spectral_vector(self, x):
        """
        Calculates the fast fourier transform of the input vector and returns the mean of the absolute values of the fft (=Amplitudes).

        Assumes dimensions to be batch_size x channels x time
        """
        fft = torch.fft.rfft(x)
        fft_abs = torch.abs(fft)
        fft_abs = fft_abs + 1e-8
        fft_abs = torch.log(fft_abs)
        fft_mean = fft_abs.mean(axis=(0, 1)).squeeze()
        return fft_mean

    def build(self, n_time: int, n_stages: int):
        blocks = nn.ModuleList()

        for stage in range(n_stages):
            blocks.append(SpectralCriticBlock(int(((n_time / 2**stage) / 2) + 1), 1))

        return blocks
