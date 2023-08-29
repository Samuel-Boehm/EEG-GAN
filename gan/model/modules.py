# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch.nn as nn
import torch
import numpy as np
from torch.nn.init import calculate_gain

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
    def apply(module, name, gain):
        fn = WeightScale(name)
        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        # Constant from He et al. 2015
        c = gain / np.sqrt(np.prod(list(weight.size())[1:]))
        setattr(module, name + '_c', float(c))
        module.register_parameter(name + '_unscaled', nn.Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))
        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_unscaled']
        del module._parameters[self.name + '_c']
        module.register_parameter(self.name, nn.Parameter(weight.data))

    def __call__(self, module, inputs, **kwargs):
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

class ConvBlockStage(nn.Module):
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
    def __init__(self, n_filters,  stage, generator=False):
        super(ConvBlockStage, self).__init__()
        kernel_size =  9 # stage0: 3, stage1: 7, stage2: 11 ....
        padding = 4 # stage0: 1, stage1: 3, stage2, 4 ...
        stride = 1 # fixed to 1 for now
        groups = int(n_filters / ((stage + 1)* 2)) # for n_filters = 120: 60, 30, 20, 15, 12, 10

        self.generator = generator

        self.conv1 = WS(nn.Conv1d(n_filters, n_filters, groups=groups, kernel_size=kernel_size, stride=stride, padding=padding))
        self.conv2 = WS(nn.Conv1d(n_filters, n_filters, groups=groups, kernel_size=kernel_size + 2, stride=stride, padding=padding + 1))
        self.conv3 = WS(nn.Conv1d(n_filters, n_filters, groups=n_filters, kernel_size=1, stride=stride, padding=0))
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.pn(x) if self.generator else x
        x = self.conv3(x)
        x = self.leaky(x)
        x = self.pn(x) if self.generator else x
        return x
    

class PrintLayer(nn.Module):
    def __init__(self, name:str):
        super(PrintLayer, self,).__init__()
        self.name = name
    def forward(self, x):
        # Do your print / debug stuff here
        print(f"####{self.name}###")
        print(x.shape)
        return x