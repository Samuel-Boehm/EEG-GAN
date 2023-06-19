import matplotlib.pyplot as plt
import numpy as np
import os
from Visualization.utils import plot_spectrum


class VisualizationHandler:
    def __init__(self, filepath, fs) -> None:
        self.filepath = filepath
        self.fs = fs

    def plot_spectrum(self, batch_real, batch_fake, epoch):
        plot_spectrum(batch_real, batch_fake, self.fs)
        plt.savefig(os.path.join(self.filepath, f"spectrum_{epoch}.png"))