import numpy as np
from torch import nn
from torch.nn.init import calculate_gain

from eeggan.examples.high_gamma.models.layers.weight_scaling import weight_scale
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

    
    def build_spectral_discriminator(self) -> ProgressiveSpectralDiscriminator:
        blocks = nn.ModuleList([])
        for i in range(self.n_stages - 1):
            in_size = int(np.floor(self.input_spectral / 2 ** i)) 
            out_size = int(np.floor(self.input_spectral / 2 ** (i + 1))) 
            blocks.append(linearBlock(in_size, out_size))

        last_block = nn.Sequential(
            nn.Linear(self.input_spectral, 1)
        )
        blocks.append(last_block)

        return ProgressiveSpectralDiscriminator(self.n_time, self.n_channels, self.n_classes, blocks)

class linearBlock(nn.Module):
    def __init__(self, in_size, out_size, use_pixelnorm=True):
        super(linearBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.lin1 = nn.Linear(in_size, out_size)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.lin1(x))
        return x

