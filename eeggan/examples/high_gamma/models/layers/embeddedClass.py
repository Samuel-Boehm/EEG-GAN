#  Author: Samuel Böhm <samuel-boehm@web.de>

import torch.nn as nn

from eeggan.utils.bias import fill_bias_zero
from eeggan.utils.weights import fill_weights_normal

class EmbeddedClassStyle(nn.Linear, nn.Module):
    def __init__(self, n_classes, n_features):
        nn.Linear.__init__(self, n_classes, n_features * 2, bias=True)
        self.n_classes = n_classes
        self.n_features = n_features
        fill_weights_normal(self.weight)
        fill_bias_zero(self.bias)

    def forward(self, x, y_onehot=None, **kwargs):
        style = nn.Linear.forward(self, y_onehot)
        style = style.view(2, x.size(0), self.n_features, 1)
        return style[0] * x + style[1]

