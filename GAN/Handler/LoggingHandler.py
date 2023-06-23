# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from torch import is_tensor
from Visualization.utils import plot_spectrum
from lightning.pytorch.callbacks import Callback
import torch
from wandb import Image




class LoggingHandler(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends.
        'pl_module' is the LightningModule and therefore can access all its methods and properties of our model
        """

        # Each epoch log: 
        trainer.logger.experiment.log({'loss generator': torch.mean(torch.cat(pl_module.loss_generator))})
        trainer.logger.experiment.log({'loss critic': torch.mean(torch.cat(pl_module.loss_critic))})

        batch_real = torch.cat(pl_module.real_data)
        batch_fake = torch.cat(pl_module.generated_data)

        # each 'plot_intervall' stages plot the spectrum of the generated and real data
        if trainer.current_epoch % pl_module.hparams.plot_interval == 0:
            # Set plotting params:
            max_freq = int(pl_module.hparams.fs / 2**(pl_module.hparams.n_stages - pl_module.current_stage))

            spectrum = self.plot_spectrum(batch_real, batch_fake, f'epoch: {self.trainer.current_epoch}',
                                     max_freq,)
            
            trainer.logger.experiment.log({'spectrum': Image(spectrum)})
        
        #TODO: critic and generator loss are empty, why?

        # Clear all variables
        pl_module.real_data.clear()
        pl_module.generated_data.clear()
        pl_module.loss_generator.clear()
        pl_module.loss_critic.clear()

       

    def plot_spectrum(self, batch_real, batch_fake, epoch, fs):

        if is_tensor(batch_real):
            batch_real = batch_real.detach().cpu().numpy()
        if is_tensor(batch_fake):
            batch_fake = batch_fake.detach().cpu().numpy()

        figure = plot_spectrum(batch_real, batch_fake, fs, epoch)
        
        return figure