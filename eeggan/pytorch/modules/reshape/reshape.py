# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

from torch import nn


class Reshape(nn.Module):
    """
    Reshape tensor into new shape

    Parameters
    ----------
    shape : list
        New shape
        Follows numpy reshaping
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        shape = list(self.shape)
        for i in range(len(shape)):
            if type(shape[i]) is list or type(shape[i]) is tuple:
                assert len(shape[i]) == 1
                shape[i] = x.size(shape[i][0])
        return x.view(shape)
