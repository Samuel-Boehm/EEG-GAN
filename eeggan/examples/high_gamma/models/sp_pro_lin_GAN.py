import numpy as np
from torch import nn
import torch
from torch.nn.init import calculate_gain

from eeggan.examples.high_gamma.models.layers.weight_scaling import weight_scale
from eeggan.training.conditional.conditionalDiscriminator import ProgressiveDiscriminatorBlock
from eeggan.training.conditional.conditionalSpectralDiscriminator import ProgressiveSpectralDiscriminator
from eeggan.examples.high_gamma.models.conditional import Conditional


class SP_GAN(Conditional):
    def __init__(self, n_stages: int, n_latent: int, n_time: int, n_channels: int,
                n_classes: int, n_filters: int, upsampling: str = 'linear',
                downsampling: str = 'linear', discfading: str = 'linear',
                genfading: str = 'linear'
                ):
    
        super().__init__(n_stages, n_latent, n_time, n_channels, n_classes, n_filters, upsampling, downsampling, discfading, genfading)
            
    def build_spectral_discriminator(self) -> ProgressiveSpectralDiscriminator:
        blocks = []
        for i in range(self.n_stages - 1):
            in_size = int(self.n_time / (2 ** (i)))
            out_size = int(self.n_time / (2 ** (i + 1)))
            block = ProgressiveDiscriminatorBlock(
                linearBlock(in_size, out_size),
                self.build_spectral_discriminator_in_sequence(i),
                self.build_disc_fade_sequence()
                )
            
            blocks.append(block)

        last_block = ProgressiveDiscriminatorBlock(
            nn.Sequential(nn.Linear(int(self.n_time / 2 ** (self.n_stages - 1)) , 1)),
            self.build_spectral_discriminator_in_sequence(self.n_stages- 1),
            None)

        blocks.append(last_block)

        return ProgressiveSpectralDiscriminator(self.n_time, self.n_channels, self.n_classes, blocks)

    def build_spectral_discriminator_in_sequence(self, stage):
    # Label Embedding is added as additional channel => self.n_channels + 1
        vector_length = self.calculate_vector_length(stage)
        in_size = vector_length
        out_size = int(self.n_time / 2 ** (stage))
        return nn.Sequential(
            weight_scale(nn.Linear(in_size, out_size), gain=calculate_gain('leaky_relu')),
            nn.LeakyReLU(0.2)
            )
    
    def calculate_vector_length(self, stage):
        n_samples = np.floor(self.n_time / 2 ** (stage))
        ### precompute indices ###
        # anticipated shift based on image size
        shift_rows = int(self.n_channels / 2) # here!!
        # number of cols after onesided fft
        cols_onesided = int(n_samples / 2) + 1
        # compute radii: shift columns by shift_y
        r = np.indices((self.n_channels  ,cols_onesided)) - np.array([[[shift_rows]],[[0]]])  # here!!
        r = np.sqrt(r[0,:,:]**2+r[1,:,:]**2)
        r = r.astype(int)
        # shift center back to (0,0)
        r = np.fft.ifftshift(r,axes=0)
        ### generate mask tensors ###
        # size of profile vector
        r_max = np.max(r)
        vector_length = r_max+1
        return vector_length

class linearBlock(nn.Module):
    def __init__(self, in_size, out_size, use_pixelnorm=True):
        super(linearBlock, self).__init__()
        self.lin1 = nn.Linear(in_size, out_size)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, **kwargs):
        x = self.leaky(self.lin1(x))
        return x
    