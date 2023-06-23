# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import matplotlib.pyplot as plt
import numpy as np
from torch import is_tensor
import os
from Visualization.utils import plot_spectrum


class VisualizationHandler:
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def plot_spectrum(self, batch_real, batch_fake, epoch, fs, filepath = None):

        if is_tensor(batch_real):
            batch_real = batch_real.detach().cpu().numpy()
        if is_tensor(batch_fake):
            batch_fake = batch_fake.detach().cpu().numpy()

        plot_spectrum(batch_real, batch_fake, fs)

        if filepath:
            plt.savefig(os.path.join(filepath, f"spectrum_{epoch}.png"))
        plt.close()