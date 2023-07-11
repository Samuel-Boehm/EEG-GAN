# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import torch
from torch import is_tensor
from lightning.pytorch.callbacks import Callback
from wandb import Image


from Visualization.utils import plot_spectrum
from GAN.Metrics.SWD import calculate_sliced_wasserstein_distance, create_wasserstein_transform_matrix


class LoggingHandler(Callback):

    def __init__(self, metric_interval:int=200):
        self.metric_interval = metric_interval

    def on_train_epoch_end(self, trainer, model):
        """
        Called when the train epoch ends.
        'pl_module' is the LightningModule and therefore can access all its methods and properties of our model
        """

        # Metrics are calculated using np arrays, so we need to convert the tensors to np arrays.
        batch_real = torch.cat(model.real_data).detach().cpu().numpy()
        batch_fake = torch.cat(model.generated_data).detach().cpu().numpy()

        # Each epoch log: 
        trainer.logger.experiment.log({'loss generator': torch.mean(torch.Tensor(model.loss_generator)),
                                       'loss critic': torch.mean(torch.Tensor(model.loss_critic)),
                                       'epoch': trainer.current_epoch,
                                       'gp': torch.mean(torch.Tensor(model.gp)),
                                       })

        # Log metrics such as FID, IS, SWD, plots etc -
        # they are more heavy to calculate, so we do not calculate them each epoch but only every metric_interval epochs.
        # a final calculation is done in the end of the training.
        if trainer.current_epoch % self.metric_interval == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            # Set plotting params:
            max_freq = int(model.hparams.fs / 2**(model.hparams.n_stages - model.current_stage))

            spectrum = plot_spectrum(batch_real, batch_fake, max_freq,
                                    f'epoch: {trainer.current_epoch}',)
            
            trainer.logger.experiment.log({'spectrum': Image(spectrum)})

            SWD, _ = self.calculate_SWD(batch_real, batch_fake)
            trainer.logger.experiment.log({'SWD': SWD})
        

        # Clear all variables
        model.real_data.clear()
        model.generated_data.clear()
        model.loss_generator.clear()
        model.loss_critic.clear()
        model.gp.clear()


    
    def calculate_FID(self,):
        pass
    
    def calculate_IS(self,):
        pass

    def calculate_SWD(self, batch_real, batch_fake):
        distances = []
        for repeat in range(10):
            self.w_transform = create_wasserstein_transform_matrix(np.prod(batch_real.shape[1:]).item())
            distances.append(calculate_sliced_wasserstein_distance(batch_real, batch_fake, self.w_transform))

        return np.mean(distances), np.std(distances)
 
    def calculate_BP(self,):
        pass