# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


def create_wasserstein_transform_matrix(
    n_features: int, n_projections: int, device: torch.device
) -> Tensor:
    """
    creates the transformation matrix for the sliced wasserstein distance
    Args:
        n_projections:   number of projections between compared distributions
        n_features:      number of features of the compared distributions
                         (e.g. number of channels in an EEG signal).
    """

    wtm = torch.randn(n_features, n_projections, device=device)
    wtm /= torch.sqrt(torch.sum(torch.square(wtm), dim=0, keepdim=True))
    return wtm


def calculate_sliced_wasserstein_distance(
    input1: Tensor, input2: Tensor, w_transform: Tensor
) -> Tensor:
    """
    calculates the sliced wasserstein distance between two distributions input1 and input2.
    Requires the transformation matrix w_transform.
    Returns the sliced wasserstein distance between the two distributions.

    Args:
        input1 (Tensor): The first distribution. Expected shape (batch_size, ...)
        input2 (Tensor): The second distribution. Expected shape (batch_size, ...)
        w_transform (Tensor): The transformation matrix for the sliced wasserstein distance,
                              created by create_wasserstein_transform_matrix.
    """
    if input1.shape[0] != input2.shape[0]:
        n_inputs = min(input1.shape[0], input2.shape[0])
        input1 = input1[
            torch.randperm(input1.shape[0], device=input1.device)[:n_inputs]
        ]
        input2 = input2[
            torch.randperm(input2.shape[0], device=input2.device)[:n_inputs]
        ]

    input1 = input1.flatten(1)
    input2 = input2.flatten(1)

    transformed1 = torch.matmul(input1, w_transform)
    transformed2 = torch.matmul(input2, w_transform)

    transformed1, _ = torch.sort(transformed1, dim=0)
    transformed2, _ = torch.sort(transformed2, dim=0)

    dists = torch.abs(transformed1 - transformed2)
    return torch.mean(dists)


class SWD(Metric):
    """
    Callback for calculating the sliced wasserstein distance between the real and generated data.
    Returns mean of `n_repetitions` using different random projections.
    """

    def __init__(self, n_projections: int = 1_000, n_repetitions: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.n_projections = n_projections
        self.n_repetitions = n_repetitions
        self.add_state("real", default=[], dist_reduce_fx="cat")
        self.add_state("fake", default=[], dist_reduce_fx="cat")

    def update(self, real: Tensor, fake: Tensor) -> None:
        if real.device.type == "cuda":
            real = real.cpu()
            fake = fake.cpu()

        self.real.append(real)
        self.fake.append(fake)

    def compute(self) -> Tensor:
        # parse inputs
        distances = []
        real = dim_zero_cat(self.real)
        fake = dim_zero_cat(self.fake)

        compute_device = real.device

        for repeat in range(self.n_repetitions):
            w_transform = create_wasserstein_transform_matrix(
                np.prod(real.shape[1:]),
                n_projections=self.n_projections,
                device=compute_device,
            )
            distances.append(
                calculate_sliced_wasserstein_distance(real, fake, w_transform)
            )

        return torch.mean(torch.stack(distances))
