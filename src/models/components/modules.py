# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import numpy as np
from torch.nn.init import calculate_gain

# For weight norm
from torch.nn.parameter import Parameter
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils import weight_norm



class WeightScale(object):
    """
    Implemented for PyTorch using WeightNorm implementation
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        w = getattr(module, self.name + '_unscaled')
        c = getattr(module, self.name + '_c')
        tmp = c * w
        return tmp
    
    @staticmethod
    def apply(module:nn.Module, name:str, gain: int)-> 'WeightScale':
        fn = WeightScale(name)
        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        # Constant from He et al. 2015
        c = gain / np.sqrt(np.prod(list(weight.size())[1:]))
        module.register_parameter(name + '_unscaled', Parameter(weight.data))
        setattr(module, name + '_c', float(c))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn


    def remove(self, module: nn.Module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_unscaled']
        del module._parameters[self.name + '_c']
        setattr(module, self.name, Parameter(weight.data))

        # module.register_parameter(self.name, nn.Parameter(weight.data))

    def __call__(self, module: nn.Module, inputs, **kwargs) -> None:
        setattr(module, self.name, self.compute_weight(module))


def WS(module, gain=calculate_gain('leaky_relu'), name='weight'):
    """
    Helper function that applies equalized learning rate to weights.
    This is only used to make the code more readable

    Parameters
    ----------
    module : module
        Module scaling should be applied to (Conv/Linear)
    gain : float
        Gain of following activation layer
        See torch.nn.init.calculate_gain. 
        Defaults to 'leaky_relu' 
    """
    WeightScale.apply(module, name, gain)
    return module


class PixelNorm(nn.Module):
    """
    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability, and Variation.
    Retrieved from http://arxiv.org/abs/1710.10196
    """

    def forward(self, x, eps=1e-8):
        tmp = torch.sqrt(torch.pow(x, 2).mean(dim=1, keepdim=True) + eps)
        return x / tmp

class ConvBlock(nn.Module):
    '''
    Description
    ----------
    Convolution block for each critic and generator stage. 
    Since the generator uses pixel norm and the critic does 
    not, the 'generator' argument can be used to toggle
    pixel norm on and off. The 'generator' argument also
    sets if the resampling layer is in the beginning or the end.

    Arguments:

    n_filters: int
        number of filters (convolution kernels) used
    stage: int
        progressive stage for which the block is build
    generator: bool
        toggle pixel norm on and off

    '''
    def __init__(self, n_filters, stage, kernel_size, is_generator=False, **kwargs):
        super(ConvBlock, self).__init__()
        stride = 1 # fixed to 1 for now
        groups = int(n_filters / ((stage)* 2)) # for n_filters = 120: 60, 30, 20, 15, 12, 10

        self.is_generator = is_generator

        n_conv_layers = 2 * stage # 2, 4, 6, 8, 10, 12

        self.convolutions = nn.ModuleList()

        for i in range(n_conv_layers):
            conv = WS(nn.Conv1d(n_filters, n_filters,
                                groups=groups,
                                kernel_size=(kernel_size + i*2),
                                stride=stride,
                                padding=1 + i))
            self.convolutions.append(conv)
        
        self.convolutions.append(nn.LeakyReLU(0.2))
        if is_generator:
            self.convolutions.append(PixelNorm())
        
        self.convolutions.append(WS(nn.Conv1d(n_filters, n_filters, groups=n_filters, kernel_size=1, stride=stride, padding=0)))
        self.convolutions.append(nn.LeakyReLU(0.2))
        
        if is_generator:
            self.convolutions.append(PixelNorm())

        if 'batch_norm' in kwargs and kwargs['batch_norm'] == True:
            self.convolutions.append(nn.BatchNorm1d(n_filters))
    
    def forward(self, x):
        
        for layer in self.convolutions:
            x = layer(x)
        return x
    

class PrintLayer(nn.Module):
    def __init__(self, name:str):
        super(PrintLayer, self,).__init__()
        self.name = name

    def forward(self, x:torch.Tensor):
        print(f"#___{self.name}___#")
        print(x.shape)
        return x