# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.visualization.stft_plots import plot_bin_stats
from gan.metrics.metric import Metric
from lightning.pytorch import Trainer, LightningModule
from gan.handler.LoggingHandler import batch_data
from wandb import Image

class BinStats(Metric):
    def __init__(self, channel_names, mapping, **kwargs):
       super().__init__(**kwargs)
       self.channel_names = channel_names
       self.mapping = mapping
    
    def __call__(self, trainer: Trainer, module: LightningModule, batch: batch_data):
        
        fs_stage = int(module.hparams.fs/2**(module.hparams.n_stages-(module.current_stage-1)))
        
        for key in self.mapping.keys():
            conditional_real = batch.real[batch.y_real == self.mapping[key]]
            conditional_fake = batch.fake[batch.y_fake == self.mapping[key]]

            # calculate frequency in current stage:   
            fig_stats, fig_real, fig_fake = plot_bin_stats(conditional_real, conditional_fake,
                            fs_stage, self.channel_names, None, str(key), False)
        
        # The image function returns a PIL image, which can be logged to wandb.
        return {f'{str(key)}_stats': Image(fig_stats),
                f'{str(key)}_real': Image(fig_real),
                f'{str(key)}_fake': Image(fig_fake)}