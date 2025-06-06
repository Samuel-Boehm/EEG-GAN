# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from typing import List

import torch
import torch.nn as nn


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

    def __init__(
        self, intermediate_sequence: nn.Sequential, out_sequence: nn.Sequential
    ) -> None:
        super().__init__()
        self.intermediate_sequence = intermediate_sequence
        self.out_sequence = out_sequence

    def forward(self, x, last=False, **kwargs):
        out = self.intermediate_sequence(x, **kwargs)
        if last:
            out = self.out_sequence(out, **kwargs)
        return out

    def stage_requires_grad(self, requires_grad: bool):
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

    def __init__(
        self,
        n_classes: int,
        latent_dim: int,
        embedding_dim: int,
        fading: bool = False,
        freeze: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.blocks: List[GeneratorBlock] = list()

        self.label_embedding = nn.Embedding(n_classes, embedding_dim)
        self.embedding_dim = embedding_dim
        self.fading = fading
        self.freeze = freeze
        self.latent_dim = latent_dim
        self.n_classes = n_classes

    def set_stage(self, cur_stage: int):
        self.cur_stage = cur_stage
        self._stage = (
            self.cur_stage - 1
        )  # Internal stage variable. Differs from GAN stage (cur_stage).

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
        # embedding shape: batch_size x 10
        x = torch.cat([x, embedding], dim=1)

        for i in range(0, self.cur_stage):
            last = i == self._stage
            if last and self.fading and self.alpha < 1:
                # if this is the last stage, fading is active and alpha < 1
                # we copy the output of the previous stage, upsample it
                # and interpolate it with the output of the current stage.
                x_ = self.blocks[i - 1].out_sequence(x, **kwargs)
                x_ = self.resample(x_, x.shape[-1] * 2)

                # pass X through last stage
                x = self.blocks[i](x, last=last, **kwargs)

                # interpolate
                x = self.alpha * x + (1 - self.alpha) * x_
            else:
                x = self.blocks[i](x, last=last, **kwargs)
        return x

    def resample(self, x: torch.Tensor, out_size: int):
        """
        rescale input. Using bicubic interpolation.

        Parameters
        ----------
        x : tensor
            Input data

        out_size : tuple

        Returns
        -------
        x : tensor
            resampled data
        """
        size = (x.shape[-2], out_size)
        x = torch.unsqueeze(x, 1)
        x = nn.functional.interpolate(x, size=size, mode="bicubic")
        x = torch.squeeze(x, 1)

        return x

    def generate(self, n_samples: int):
        z = torch.randn(n_samples, self.latent_dim)

        y_fake = torch.randint(
            low=0, high=self.n_classes, size=(n_samples,), dtype=torch.int32
        )

        z, y_fake = self._to_current_device([z, y_fake])

        X_fake = self.forward(z, y_fake)

        return X_fake, y_fake

    def _to_current_device(self, inputs: List[torch.Tensor]):
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
