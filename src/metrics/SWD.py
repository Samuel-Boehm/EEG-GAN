# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import torch
from lightning.pytorch import Trainer, LightningModule
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from torch import Tensor

def create_wasserstein_transform_matrix(n_features:int, n_projections:int, device) -> Tensor:
    """
    creates the transformation matrix for the sliced wasserstein distance
    Args:
        n_projections:  number of projections between compared distributions
        n_features:     number of features of the compared distributions
                        (e.g. number of channels in an EEG signal). Default: 100
    """


    wtm = torch.randn(n_features, n_projections)
    wtm /= torch.sqrt(torch.sum(torch.square(wtm), axis=0, keepdims=True))
    return wtm.to(device)


def calculate_sliced_wasserstein_distance(input1:Tensor, input2:Tensor, w_transform:Tensor) -> float:
    """
    calculates the sliced wasserstein distance between two distributions input1 and input2. 
    Requires the transformation matrix w_transform.
    Returns the sliced wasserstein distance between the two distributions.
    """
    if input1.shape[0] != input2.shape[0]:
        n_inputs = input1.shape[0] if input1.shape[0] < input2.shape[0] else input2.shape[0]
        input1 = torch.randperm(input1)[:n_inputs]
        input2 = torch.randperm(input2)[:n_inputs]

    input1 = input1.reshape(input1.shape[0], -1)
    input2 = input2.reshape(input2.shape[0], -1)

    transformed1 = torch.matmul(input1, w_transform)
    transformed2 = torch.matmul(input2, w_transform)

    transformed1, _ = torch.sort(transformed1, axis=0)                                  
    transformed2, _ = torch.sort(transformed2, axis=0)

    dists = torch.abs(transformed1 - transformed2)
    return torch.mean(dists)


class SWD(Metric):
    """ 
    Callback for calculating the sliced wasserstein distance between the real and generated data. 
    Returns mean of 10 repetitions using different random projections.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("real", default=[], dist_reduce_fx="cat")
        self.add_state("fake", default=[], dist_reduce_fx="cat")
    
    def update(self, real: Tensor, fake: Tensor) -> None:
        # If real and fake are on the GPU, we need to move them to the CPU
        if real.device.type == 'cuda':
            real = real.cpu()
            fake = fake.cpu()

        self.real.append(real)
        self.fake.append(fake)

    def compute(self) -> Tensor:
        # parse inputs
        distances = []
        real = dim_zero_cat(self.real)
        fake = dim_zero_cat(self.fake)
        for repeat in range(10):
            w_transform = create_wasserstein_transform_matrix(np.prod(real.shape[1:]), n_projections=1_000, device=real.device)
            distances.append(calculate_sliced_wasserstein_distance(real, fake, w_transform))

        return torch.mean(torch.stack(distances))