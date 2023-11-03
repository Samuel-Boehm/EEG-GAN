# Author: Kay Hartmann <kg.hartma@gmail.com>
# Project: EEG-GAN

from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from gan.metrics.metric import Metric



def calculate_inception_score(preds: Tensor, splits: int = 1, repititions: int = 1) -> Tuple[float, float]:
    with torch.no_grad():
        stepsize = np.max((int(np.ceil(preds.size(0) / splits)), 2))
        steps = np.arange(0, preds.size(0), stepsize)
        scores = []
        for rep in np.arange(repititions):
            preds_tmp = preds[torch.randperm(preds.size(0), device=preds.device)]
            if len(preds_tmp) < 2:
                continue
            for i in np.arange(len(steps)):
                preds_step = preds_tmp[steps[i]:steps[i] + stepsize]
                step_mean = torch.nanmean(preds_step, 0, keepdim=True)
                kl = preds_step * (torch.log(preds_step) - torch.log(step_mean))
                kl = torch.nanmean(torch.sum(kl, 1))
                scores.append(torch.exp(kl).item())

        return np.nanmean(scores).item(), np.nanstd(scores).item()
    

class FrechetMetric(Metric):

    def __init__(self, deep4s: List[Module], *args, **kwargs):

        self.deep4s = deep4s
        super().__init__(*args, **kwargs)

    def __call__(self, trainer: Trainer, module: LightningModule, batch: batch_data) -> Tuple[float, float]:

        r"""
        Returns the mean and standard deviation of the Frechet distance between real and fake data.
        """

        with torch.no_grad():
            X_real = Tensor(batch.real)
            X_real = X_real[:100, :, :, None]
            X_real = X_real.to(device='cuda')


            X_fake = Tensor(batch.real)
            X_fake = X_fake[100:200, :, :, None]
            X_fake = X_fake.to(device='cuda')

            dists = []

            for deep4 in self.deep4s:
                mu_real, sig_real = calculate_activation_statistics(deep4(X_real)[0])
                mu_fake, sig_fake = calculate_activation_statistics(deep4(X_fake)[0])
                dist = calculate_frechet_distances(mu_real[None, :, :], sig_real[None, :, :], mu_fake[None, :, :],
                                                   sig_fake[None, :, :]).item()
                dists.append(dist)

            return np.mean(dists).item(), np.std(dists).item()