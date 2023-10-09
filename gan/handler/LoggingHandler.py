# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer, LightningModule
from wandb import Image
from matplotlib.pyplot import close as close_figures
from gan.data.batch import batch_data
from gan.metrics.metric import Metric
import torch


class LoggingHandler(Callback):

    def __init__(self):
        self.running_metrics = []
        self.end_metrics = []

    def attach_metrics(self, metrics:list,):
        """
        Attach metrics to the callback.
        Args:
            metrics: list of metrics to attach to the callback.
            every_epoch: if True the metrics are calculated at the end of each epoch.
        """
        for metric in metrics:
            if not isinstance(metric, Metric):
                raise TypeError(f'Expected metric to be of type Metric, but got {type(metric)}')
            if metric.interval == 0:
                self.end_metrics.append(metric)
            else:
                self.running_metrics.append(metric)
    

    def clear(self, module:LightningModule):
        """ 
        After each epoch, the lists in the module are cleared.
        """
        # Clear all variables
        module.real_data.clear()
        module.generated_data.clear()
        module.y_real.clear()
        module.y_fake.clear()
        module.loss_generator.clear()
        module.loss_critic.clear()
        module.gp.clear()

        # close all figures
        close_figures('all')


    def on_train_epoch_end(self, trainer:Trainer, module:LightningModule):
        """
        Called when the train epoch ends.
        Args:
            trainer: the trainer object. Used to access the trainer class.
            model: the model object. Used to access the model class.
        """

        # Metrics are calculated using np arrays, so we need to convert the tensors to np arrays.
        X_real = np.concatenate(module.real_data)
        X_fake = np.concatenate(module.generated_data)

        y_real = np.concatenate(module.y_real)
        y_fake = np.concatenate(module.y_fake)

        batch = batch_data(X_real, X_fake, y_real, y_fake)

        # Calculate running metrics:
        for metric in self.running_metrics:
            if trainer.current_epoch % metric.interval == 0:
                trainer.logger.experiment.log(metric(trainer, module, batch))
        
        # Each epoch log: 
        trainer.logger.experiment.log({'loss generator': np.mean(module.loss_generator),
                                       'loss critic': np.mean(module.loss_critic),
                                       'epoch': trainer.current_epoch,
                                       'gp': np.mean(module.gp),
                                    })
        
        # Calculate end metrics:
        if trainer.current_epoch == trainer.max_epochs - 1:
            for metric in self.end_metrics:
                trainer.logger.experiment.log(metric(trainer, module, batch))

            for metric in self.running_metrics:
                trainer.logger.experiment.log(metric(trainer, module, batch))
        
        self.clear(module)

    

        
        