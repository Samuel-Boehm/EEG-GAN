# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np

def create_wasserstein_transform_matrix(n_features, n_projections:int=100):
    ''' creates the transformation matrix for the sliced wasserstein distance
    Args:
        n_projections:  number of projections between copared distributions
        n_features:     number of features of the compared distributions
                        (e.g. number of channels in an EEG signal). Default: 100
    '''


    wtm = np.random.randn(n_features, n_projections)
    wtm /= np.sqrt(np.sum(np.square(wtm), axis=0, keepdims=True))
    return wtm


def calculate_sliced_wasserstein_distance(input1, input2, w_transform):
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