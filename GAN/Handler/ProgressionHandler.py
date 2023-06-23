# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from torch import is_tensor
from Visualization.utils import plot_spectrum
from lightning.pytorch.callbacks import Callback
import torch
from wandb import Image


class Scheduler(Callback):
    
    def on_train_epoch_start(self, trainer, model):
        '''
        Each epoch start, check if we need to increase stage. If so, increase stage.
        '''
        
        if trainer.current_epoch in model.progression_epochs:
            # set stage in data, critic and generator
            trainer.datamodule.set_stage(model.current_stage)
            model.generator.set_stage(model.current_stage)
            model.critic.set_stage(model.current_stage)

            # increase stage
            model.current_stage += 1 
