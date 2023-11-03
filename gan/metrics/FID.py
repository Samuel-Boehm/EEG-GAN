# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

# Code was taken from https://github.com/AliaksandrSiarohin/gan/blob/master/fid.py 
# and modified to work on EEG data using the Deep4 model from braindecode.

import numpy as np
from typing import List, Tuple
from lightning.pytorch import Trainer, LightningModule

import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader

from gan.data.batch import batch_data
from gan.metrics.metric import Metric
from gan.utils import to_device


def calculate_activation_statistics(act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        act = act.reshape(act.shape[0], -1)
        fact = act.shape[0] - 1
        mu = torch.mean(act, dim=0, keepdim=True)
        act = act - mu.expand_as(act)
        sigma = act.t().mm(act) / fact
        return mu, sigma


# From https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/frechet_inception_distance.py
def _calculate_FID(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor,
                                sigma2: torch.Tensor) -> torch.Tensor:
    r"""
    Numpy implementation of the Frechet Distance betweem activations.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
    Returns:
    -- dist  : The Frechet Distance.
    Raises:
    -- InvalidFIDException if nan occures.
    """
    with torch.no_grad():
        m = torch.square(mu1 - mu2).sum()
        d = torch.bmm(sigma1, sigma2)
        s = sqrtm_newton(d)
        dists = m + torch.diagonal(sigma1 + sigma2 - 2 * s, dim1=-2, dim2=-1).sum(-1)
        return dists


# https://colab.research.google.com/drive/1wSO1MFh_ZCfOnejFnW1vkD71jaJy2Olu#scrollTo=Ju79uoiTQku6&line=1&uniqifier=1
def sqrtm_newton(A: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        numIters = 20
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(A.dtype).to(A)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
        return sA
    
def calculate_FID(real: torch.Tensor, fake: torch.Tensor,
                  deep4s: Module, device = 'cuda') -> Tuple[float, float]:
    r'''
    Calculates the Frechet Inception Distance (FID) between two batches of data.
    Args:
        real: The real data.
        fake: The fake data.
        deep4s: A list of Deep4 models. Note: to access the intermediate layers, 
                the Deep4 model needs to be wrapped with the IntermediateOutputWrapper.
    '''
    with torch.no_grad():
        real = real[:, :, :, None]
        fake = fake[:, :, :, None]
        to_device(device, real, fake,)
        real_dl = DataLoader(real, batch_size=128, num_workers=2,)
        fake_dl = DataLoader(fake, batch_size=128, num_workers=2,)

        dists = []
        for deep4 in deep4s:
            for real_batch, fake_batch in zip(real_dl, fake_dl):
                mu_real, sig_real = calculate_activation_statistics(deep4(real_batch)[0])
                mu_fake, sig_fake = calculate_activation_statistics(deep4(fake_batch)[0])
                dist = _calculate_FID(mu_real[None, :, :], sig_real[None, :, :],
                                      mu_fake[None, :, :], sig_fake[None, :, :]).item()
                dists.append(dist)
        print(len(dists), 'number of distances calculated.')
        return np.mean(dists).item(), np.std(dists).item()


class FrechetMetric(Metric):

    def __init__(self, deep4s: List[Module], *args, **kwargs):

        self.deep4s = deep4s
        super().__init__(*args, **kwargs)

    def __call__(self, trainer: Trainer, module: LightningModule, batch: batch_data) -> Tuple[float, float]:

        r"""
        Returns the mean and standard deviation of the Frechet distance between real and fake data.
        """

        mean, std = calculate_FID(batch.real, batch.fake, self.deep4s)

        return mean, std

        