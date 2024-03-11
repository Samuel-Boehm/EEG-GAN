# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
from gan.metrics.metric import Metric
from lightning.pytorch import Trainer, LightningModule
from gan.handler.logginghandler import batch_data


def create_wasserstein_transform_matrix(n_features, n_projections:int=100):
    """
    creates the transformation matrix for the sliced wasserstein distance
    Args:
        n_projections:  number of projections between compared distributions
        n_features:     number of features of the compared distributions
                        (e.g. number of channels in an EEG signal). Default: 100
    """


    wtm = np.random.randn(n_features, n_projections)
    wtm /= np.sqrt(np.sum(np.square(wtm), axis=0, keepdims=True))
    return wtm


def calculate_sliced_wasserstein_distance(input1, input2, w_transform):
    """
    calculates the sliced wasserstein distance between two distributions input1 and input2. 
    Requires the transformation matrix w_transform.
    Returns the sliced wasserstein distance between the two distributions.
    """
    if input1.shape[0] != input2.shape[0]:
        n_inputs = input1.shape[0] if input1.shape[0] < input2.shape[0] else input2.shape[0]
        input1 = np.random.permutation(input1)[:n_inputs]
        input2 = np.random.permutation(input2)[:n_inputs]

    input1 = input1.reshape(input1.shape[0], -1)
    input2 = input2.reshape(input2.shape[0], -1)

    transformed1 = np.matmul(input1, w_transform)
    transformed2 = np.matmul(input2, w_transform)

    transformed1 = np.sort(transformed1, axis=0)                                  
    transformed2 = np.sort(transformed2, axis=0)

    dists = np.abs(transformed1 - transformed2)
    return np.mean(dists)


class SWD(Metric):
    """ 
    Callback for calculating the sliced wasserstein distance between the real and generated data. 
    Returns mean of 10 repetitions using different random projections.
    """

    def __call__(self, trainer: Trainer, pl_module: LightningModule, batch: batch_data):
        distances = []
        for repeat in range(10):
            self.w_transform = create_wasserstein_transform_matrix(np.prod(batch.real.shape[1:]).item())
            distances.append(calculate_sliced_wasserstein_distance(batch.real, batch.fake, self.w_transform))

        return {'SWD' : np.mean(distances)}