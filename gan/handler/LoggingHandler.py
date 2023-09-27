# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import torch
from torch import is_tensor
from lightning.pytorch.callbacks import Callback
from wandb import Image
from matplotlib.pyplot import close as close_figures
import os



from visualization.utils import plot_spectrum
from gan.metrics.SWD import calculate_sliced_wasserstein_distance, create_wasserstein_transform_matrix
from gan.visualization.stft_plots import plot_bin_stats

class LoggingHandler(Callback):

    def __init__(self, metric_interval:int=200, channel_names:list=None,):
        self.metric_interval = metric_interval
        self.channel_names = channel_names

    def on_train_epoch_end(self, trainer, model):
        """
        Called when the train epoch ends.
        'pl_module' is the LightningModule and therefore can access all its methods and properties of our model
        """

        # Metrics are calculated using np arrays, so we need to convert the tensors to np arrays.
        batch_real = np.concatenate(model.real_data)
        batch_fake = np.concatenate(model.generated_data)

        y_real = np.concatenate(model.y_real)
        y_fake = np.concatenate(model.y_fake)

        SWD, _ = self.calculate_SWD(batch_real, batch_fake)

        # Each epoch log: 
        trainer.logger.experiment.log({'loss generator': np.mean(model.loss_generator),
                                       'loss critic': np.mean(model.loss_critic),
                                       'epoch': trainer.current_epoch,
                                       'gp': np.mean(model.gp),
                                       'SWD': SWD,
                                       'fd loss': np.mean(model.fd_loss),
                                       })
        

        # Log metrics such as FID, IS, plots etc -
        # they are more heavy to calculate, so we do not calculate them each epoch but only every metric_interval epochs.
        # a final calculation is done in the end of the training.
        if trainer.current_epoch % self.metric_interval == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            # Set plotting params:
            max_freq = int(model.hparams.fs / 2**(model.hparams.n_stages - model.current_stage))

            spectrum = plot_spectrum(batch_real, batch_fake, max_freq,
                                        f'epoch: {trainer.current_epoch}',)
            
            trainer.logger.experiment.log({'spectrum': Image(spectrum)})


        # Plots such as the bin percentage, and the time domain plot are only calculated at stage end.
        # Also we will collect some statistical measures here.
        # !!!!!!!! Attention !!!!!!!! 
        # This part is still work in progress and not finished yet.
        # hardcoded the mapping for now

        if trainer.current_epoch % self.metric_interval == 0 or trainer.current_epoch == trainer.max_epochs - 1:
            MAPPING = {'right': 0, 'rest': 1}
            for key in MAPPING.keys():
                conditional_real = batch_real[y_real == MAPPING[key]]
                conditional_fake = batch_fake[y_fake == MAPPING[key]]

                # calculate frequency in current stage: 
                fs_stage = int(model.hparams.fs / 2**(model.hparams.n_stages - model.current_stage))
                
                fig_stats, fig_real, fig_fake = plot_bin_stats(conditional_real, conditional_fake,
                                fs_stage, self.channel_names, None, str(key), False)
                
                trainer.logger.experiment.log({f'{str(key)}_stats': Image(fig_stats)})
                trainer.logger.experiment.log({f'{str(key)}_real': Image(fig_real)})
                trainer.logger.experiment.log({f'{str(key)}_fake': Image(fig_fake)})
        
        # Clear all variables
        model.real_data.clear()
        model.generated_data.clear()
        model.y_real.clear()
        model.y_fake.clear()
        model.loss_generator.clear()
        model.loss_critic.clear()
        model.gp.clear()

        # close all figures
        close_figures('all')


    
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