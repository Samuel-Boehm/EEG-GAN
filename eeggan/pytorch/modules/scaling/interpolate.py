#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import Iterable, Union

from torch import Tensor, nn

from eeggan.pytorch.modules.module import Module


class Interpolate(Module):
    """
    Interpolation layer. Down/up samples the input to the given scale_factor. 
    Mode sets the used algorithm for interpolation. 
    """
    def __init__(self, scale_factor: Union[float, Iterable[float]], mode: str):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return nn.functional.interpolate(x, size=None, scale_factor=self.scale_factor, mode=self.mode)
