#  Author:  Kay Hartmann <kg.hartma@gmail.com>
#           Samuel Böhm <samuel-boehm@web.de>

from abc import ABC, abstractmethod
from torch import nn
from eeggan.model.discriminator import Discriminator
from eeggan.model.generator import Generator
from eeggan.model.progressive.discriminator import ProgressiveDiscriminator
from eeggan.model.progressive.generator import ProgressiveGenerator


class ModelBuilder(ABC):
    """
    Baseclass to build the EEG-GAN Model
    """
    @abstractmethod
    def build_discriminator(self) -> Discriminator:
        raise NotImplementedError

    @abstractmethod
    def build_generator(self) -> Generator:
        raise NotImplementedError


class ProgressiveModelBuilder(ModelBuilder, ABC):
    """Baseclass to build a progressive GAN Model"""

    def __init__(self, n_stages: int):
        self.n_stages = n_stages

    @abstractmethod
    def build_disc_conv_sequence(self, i_stage: int) -> nn.Sequential:
        raise NotImplementedError
    
    @abstractmethod
    def build_disc_in_sequence(self) -> nn.Sequential:
        raise NotImplementedError
    
    @abstractmethod
    def build_disc_fade_sequence(self) -> nn.Sequential:
        raise NotImplementedError
    
    @abstractmethod
    def build_discriminator(self) -> ProgressiveDiscriminator:
        raise NotImplementedError

    @abstractmethod
    def build_gen_conv_sequence(self, i_stage: int) -> nn.Sequential:
        raise NotImplementedError

    @abstractmethod
    def build_gen_out_sequence(self) -> nn.Sequential:
        raise NotImplementedError

    @abstractmethod
    def build_gen_fade_sequence(self) -> nn.Sequential:
        raise NotImplementedError

    @abstractmethod
    def build_generator(self) -> ProgressiveGenerator:
        raise NotImplementedError
