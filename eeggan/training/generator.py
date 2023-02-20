#  Author: Kay Hartmann <kg.hartma@gmail.com>

from abc import ABCMeta

from numpy.random.mtrand import RandomState
from torch import Tensor
from typing import Tuple
import torch

from eeggan.data.preprocess.util import create_onehot_vector
from eeggan.pytorch.modules.module import Module


class Generator(Module, metaclass=ABCMeta):
    """
    Base Generator Class
    """
    def __init__(self, n_samples, n_channels, n_classes, n_latent):
        super().__init__()
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_latent = n_latent

    def create_latent_input(self, rng: RandomState, n_trials, balanced = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Create latent input for generator
        
        Returns:
            Tensor: z_latent, y_fake y_fake_onehot

        """

        if balanced:
            assert n_trials % self.n_classes == 0, (
                f'Can not create balanced input for n_trials = {n_trials} and n_classes = {self.n_classes} since {n_trials} % {self.n_classes} needs to be 0 ')
            z_latent = rng.normal(0, 1, size=(n_trials, self.n_latent))
            y_fake = torch.arange(self.n_classes).tile((int(n_trials/self.n_classes),))
            y_fake_onehot = create_onehot_vector(y_fake, self.n_classes)
        else:
            z_latent = rng.normal(0, 1, size=(n_trials, self.n_latent))
            y_fake = rng.randint(0, self.n_classes, size=n_trials)
            y_fake_onehot = create_onehot_vector(y_fake, self.n_classes)
        return Tensor(z_latent), Tensor(y_fake), Tensor(y_fake_onehot)

