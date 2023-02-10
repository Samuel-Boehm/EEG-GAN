#  Author: Kay Hartmann <kg.hartma@gmail.com>

import os
from abc import ABCMeta
import numpy as np

from matplotlib.figure import Figure

from eeggan.plotting.plots import spectral_plot, labeled_tube_plot
from eeggan.training.trainer.trainer import Trainer, BatchOutput
from eeggan.training.trainer.gan_softplus_spectral import SpectralTrainer

class EpochPlot(metaclass=ABCMeta):
    def __init__(self, figure: Figure, plot_path: str, prefix: str, tb_writer=None):
        self.writer = tb_writer # Tensorboard writer
        self.figure = figure
        self.path = plot_path
        self.prefix = prefix

    def __call__(self, trainer: Trainer):
        self.plot(trainer)
        self.figure.savefig(os.path.join(self.path, self.prefix + str(trainer.state.epoch)))
        if self.writer:
            self.writer.add_figure(self.prefix + str(trainer.state.epoch), self.figure)
        self.figure.clear()

    def plot(self, trainer: Trainer):
        raise NotImplementedError


class SpectralPlot(EpochPlot):
    def __init__(self, figure: Figure, plot_path: str, prefix: str, n_samples: int, fs: float, tb_writer=None):
        self.n_samples = n_samples
        self.fs = fs
        super().__init__(figure, plot_path, prefix, tb_writer)

    def plot(self, trainer: Trainer):
        batch_output: BatchOutput = trainer.state.output
        spectral_plot(batch_output.batch_real.X.data.cpu().numpy(), batch_output.batch_fake.X.data.cpu().numpy(),
                      self.fs, self.figure.gca())
        

class DiscriminatorSpectrum(EpochPlot):

    def __init__(self, figure: Figure, plot_path: str, prefix: str, tb_writer=None):
        super().__init__(figure, plot_path, prefix, tb_writer)

    def plot(self, trainer: SpectralTrainer):
        batch_output: BatchOutput = trainer.state.output

        svr = trainer.spectral_discriminator.get_input(batch_output.batch_real.X, batch_output.batch_real.y)
        svf = trainer.spectral_discriminator.get_input(batch_output.batch_fake.X, batch_output.batch_fake.y)
        
        svr = svr.detach().cpu().numpy()
        svf = svf.detach().cpu().numpy()
        
        mean_r = np.mean(svr, axis=0)
        std_r = np.std(svr, axis=0)

        mean_f = np.mean(svf, axis=0)
        std_f = np.std(svf, axis=0)
        
        x = np.arange(len(mean_r))

        labeled_tube_plot(x,
                      [mean_r, mean_f],
                      [std_r, std_f],
                      ["Real", "Fake"],
                      "Spectral Vector", "Hz", "Normalized vector", self.figure.gca())

