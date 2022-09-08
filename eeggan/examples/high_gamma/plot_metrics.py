#  Author: Kay Hartmann <kg.hartma@gmail.com>
import os
from this import d

import sys
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

import numpy as np
import torch
import matplotlib.pyplot as plt

from eeggan.plotting.plots import labeled_plot, labeled_tube_plot

SUBJ_IND = 1
RESULT_PATH = '/home/boehms/eeg-gan/EEG-GAN/Data/Results/baseline/%d/' % SUBJ_IND
STAGE = 0

if __name__ == '__main__':
    metrics = torch.load(os.path.join(RESULT_PATH, 'metrics_stage_%d.pt' % STAGE))
    metric_wasserstein = np.asarray(metrics["wasserstein"], dtype=object)
    metric_inception = np.asarray(metrics["inception"], dtype=object)
    metric_frechet = np.asarray(metrics["frechet"], dtype=object)
    metric_loss = metrics["loss"]
    metric_classification = np.asarray(metrics["classification"], dtype=object)

    fig = plt.figure()
    labeled_plot(
        metric_wasserstein[:, 0],
        [metric_wasserstein[:, 1]],
        ["fake to train"],
        "Sliced Wasserstein Distance", "Epochs", "SWD", fig.gca()
    )
    plt.show()

    fig = plt.figure()
    labeled_tube_plot(
        metric_inception[:, 0].astype(float),
        [np.asarray([np.asarray(t, dtype=float) for t in metric_inception[:, 1]], dtype=float)[:, 0]],
        [np.asarray([np.asarray(t, dtype=float) for t in metric_inception[:, 1]], dtype=float)[:, 1]],
        ["fake"],
        "Inception scores", "Epochs", "score", fig.gca()
    )
    plt.show()

    fig = plt.figure()
    labeled_tube_plot(
        metric_frechet[:, 0].astype(float),
        [np.asarray([np.asarray(t, dtype=float) for t in metric_frechet[:, 1]], dtype=float)[:, 0]],
        [np.asarray([np.asarray(t, dtype=float) for t in metric_frechet[:, 1]], dtype=float)[:, 1]],
        ["fake"],
        "Frechet Distance", "Epochs", "distance", fig.gca()
    )
    plt.show()

    fig = plt.figure()
    labeled_tube_plot(
        metric_classification[:, 0].astype(float),
        [np.asarray([np.asarray(t, dtype=float) for t in metric_classification[:, 1]], dtype=float)[:, 0]],
        [np.asarray([np.asarray(t, dtype=float) for t in metric_classification[:, 1]], dtype=float)[:, 1]],
        ["fake"],
        "Classification Accuracy", "Epochs", "accuracy %", fig.gca()
    )
    plt.show()
