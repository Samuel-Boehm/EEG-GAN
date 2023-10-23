# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from typing import Any
from gan.metrics.metric import Metric
from lightning.pytorch import Trainer, LightningModule
from gan.data.batch import batch_data
from wandb import Image
from gan.visualization.utils import plot_spectrum

class Spectrum(Metric):

    def __call__(self, trainer: Trainer, module: LightningModule, batch: batch_data) -> Any:

        fs_stage = int(module.hparams.fs/2**(module.hparams.n_stages-(module.current_stage-1)))

        spectrum = plot_spectrum(batch.real, batch.fake, fs_stage, f'epoch: {trainer.current_epoch}',)

        return {'spectrum': Image(spectrum)}
        