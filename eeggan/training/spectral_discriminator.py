#  Author: Samuel Böhm <samuel-boehm@web.de>

import torch.nn as nn
import torch
import numpy as np

from eeggan.training.discriminator import Discriminator
from eeggan.training.spectralLoss import SpectralLoss
from eeggan.pytorch.modules.module import Module
from eeggan.pytorch.modules.sequential import Sequential



#  Author: Samuel Böhm <samuel-boehm@web.de>


class Unnormalize(Module):
    
    ###########################################################################
    def __init__(self):
        super(Unnormalize, self).__init__()
        
    ###########################################################################
    def forward(self, input):
        return (input + 1) / 2
    
##############################################################################

class Normalize(Module):
    
    ###########################################################################
    def __init__(self):
        super(Normalize, self).__init__()
        
    ###########################################################################
    def forward(self, input):
        return (input - 0.5) / 0.5



class SpectralDiscriminator(Discriminator):
    
    ###########################################################################
    def __init__(self, n_samples, n_channels, n_classes,  spectral = "linear"):
        super().__init__(n_samples, n_channels, n_classes)
        
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.spectral = spectral

        self.spectral_transform = SpectralLoss(rows=n_channels, cols=n_samples)

        self._add_spectral_layers(spectral)
        
    ###########################################################################
    def _add_spectral_layers(self, spectral):
        if spectral == "none":
            self.forward = self.forward_none

        else:
            layers = Sequential()
            
            if "unnormalize" in spectral:
                layers.add_module("Unnormalize", Unnormalize())
                
            if "dropout" in spectral:
                layers.add_module("Dropout", nn.Dropout())
                
            if "linear" in spectral and not "nonlinear" in spectral:
                layers.add_module("LinearSpectral", nn.Linear(self.spectral_transform.vector_length, 1))
    
            if "nonlinear" in spectral:
                layers.add_module("Linear1Spectral", nn.Linear(self.spectral_transform.vector_length, self.spectral_transform.vector_length))
                layers.add_module("ReLU1Spectral",   nn.LeakyReLU(0.2))
                layers.add_module("Linear1Spectral", nn.Linear(self.spectral_transform.vector_length, self.spectral_transform.vector_length))
                layers.add_module("ReLU1Spectral",   nn.LeakyReLU(0.2))
                layers.add_module("Linear1Spectral", nn.Linear(self.spectral_transform.vector_length, self.spectral_transform.vector_length))
                layers.add_module("ReLU1Spectral",   nn.LeakyReLU(0.2))
                layers.add_module("Linear2Spectral", nn.Linear(self.spectral_transform.vector_length, 1))
                
            self._forward_spectral = layers
            
    ###########################################################################
    def forward(self, x, **kwargs):
        x_profiles = self.spectral_transform.spectral_vector(x, **kwargs)
        y = self._forward_spectral(x_profiles)        
        return y
    
    ###########################################################################
    def forward_none(self, x):
        return torch.tensor(0.0)
    
    ###########################################################################
    def par_count(self):
        c = 0
        for p in self.parameters():
            c += np.prod(p.shape)
        return c
    
    ###########################################################################
    def print_par_count(self):
        for name, p in self.named_parameters():
            print(f"{name:>40}: {str(p.shape):>40} {np.prod(p.shape):>15,}")
    
    ###########################################################################
    def load(self, state):
        self.load_state_dict(state)
            
    ###########################################################################
    def to_checkpoint(self):
        chkpt = {}
        chkpt["state"]   = self.state_dict()
        chkpt["pars"] = {
            "img_size"   : self.img_size,
            "spectral"   : self.spectral,
        }
        return chkpt

    ###########################################################################
    @staticmethod
    def from_checkpoint(chkpt):
        D = Discriminator(**chkpt["pars"])
        D.load(chkpt["state"])
        return D
