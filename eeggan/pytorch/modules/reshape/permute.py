# coding=utf-8

#  Author: Kay Hartmann <kg.hartma@gmail.com>

from torch import nn


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x, **kwargs):
        return x.permute(*self.dims)
