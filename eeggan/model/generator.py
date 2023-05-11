#  Author: Samuel Böhm <samuel-boehm@web.de>

from abc import ABC, abstractmethod

from numpy.random.mtrand import RandomState
from torch import Tensor
from typing import Tuple
import torch
import pytorch_lightning as pl

class Generator(pl.LightningModule, ABC):
    """
    Base Generator Class
    """
    def __init__(self, n_samples, n_channels, n_classes, n_latent):
        super().__init__()
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_latent = n_latent

    def create_latent_input(self, rng: RandomState, n_trials, balanced = False) -> Tuple[Tensor, Tensor]:
        """
        Create latent input for generator
        Parameters:
            rng: RandomState
            n_trials (int): defines how many vectors are returned respectively how many samples are generated
            balanced (bool): If True the returned labels 'y_fake' are balanced

        
        Returns:
            Tensor: z_latent, y_fake

        """

        if balanced:
            assert n_trials % self.n_classes == 0, (
                f'Can not create balanced input for n_trials = {n_trials} and \
                n_classes = {self.n_classes} since {n_trials} % {self.n_classes} needs to be 0 ')
            
            z_latent = rng.normal(0, 1, size=(n_trials, self.n_latent))
            y_fake = torch.arange(self.n_classes).tile((int(n_trials/self.n_classes),))
        else:
            z_latent = rng.normal(0, 1, size=(n_trials, self.n_latent))
            y_fake = rng.randint(0, self.n_classes, size=n_trials)
        return Tensor(z_latent), Tensor(y_fake)

    @abstractmethod
    def forward(self):
        pass
