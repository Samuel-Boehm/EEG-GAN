import numpy as np
from torch import nn
from torch.nn.init import calculate_gain

from eeggan.pytorch.modules.reshape.reshape import Reshape
from eeggan.pytorch.modules.sequential import Sequential
from eeggan.pytorch.modules.weights.weight_scaling import weight_scale
from eeggan.training.conditional.conditionalDiscriminator import ProgressiveDiscriminatorBlock
from eeggan.training.conditional.conditionalSpectralDiscriminator import ProgressiveSpectralDiscriminator
from eeggan.examples.high_gamma.models.conditional import Conditional

class SP_GAN(Conditional):
    def __init__(self, n_stages: int, n_latent: int, n_time: int, n_channels: int,
    n_classes: int, n_filters: int, upsampling: str = 'linear', downsampling: str = 'linear',
    discfading: str = 'linear', genfading: str = 'linear'):
    
        super().__init__(n_stages, n_latent, n_time, n_channels, n_classes, n_filters, upsampling, downsampling, discfading, genfading)
        
        self.input_spectral = int(n_time / 2) + 1
        self.n_time_last_layer_spectral = int(np.floor(self.input_spectral / 2 ** n_stages)) 

        print(self.n_time_last_layer, self.n_time_last_layer_spectral)
    
    def build_spectral_discriminator(self) -> ProgressiveSpectralDiscriminator:
        blocks = []
        for i in range(self.n_stages - 1):
            block = ProgressiveDiscriminatorBlock(
                self.build_disc_conv_sequence(self.n_stages - 1 - i),
                self.build_disc_in_sequence(),
                self.build_disc_fade_sequence()
            )
            blocks.append(block)

        last_block = ProgressiveDiscriminatorBlock(
            Sequential(
                self.build_disc_conv_sequence(0),
                Reshape([[0], self.n_filters * self.n_time_last_layer_spectral]),
                weight_scale(nn.Linear(self.n_filters * self.n_time_last_layer_spectral, 1),
                             gain=calculate_gain('linear'))
            ),
            self.build_disc_in_sequence(),
            None
        )
        blocks.append(last_block)
        return ProgressiveSpectralDiscriminator(self.n_time, self.n_channels, self.n_classes, blocks)