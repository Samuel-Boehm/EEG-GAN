#  Authors: Kay Hartmann <kg.hartma@gmail.com>
#           Samuel Böhm <samuel-boehm@web.de>

import numpy as np
import ot



def create_wasserstein_transform_matrix(n_projections, n_features):
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

def calculated_wasserstein_distance_POT(input1,input2, n_projections):

    input1 = input1.reshape(input1.shape[0], -1)
    input2 = input2.reshape(input2.shape[0], -1)

    return ot.sliced_wasserstein_distance(input1, input2, n_projections=n_projections).item()