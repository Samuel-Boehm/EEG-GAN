# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.GAN import GAN
from lightning import Trainer
import torch
import os
from gan.Data.datamodule import HighGammaModule as HDG
from gan.handler.ProgressionHandler import Scheduler
from gan.handler.LoggingHandler import LoggingHandler
from gan.paths import data_path, results_path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Import metrics:
from gan.metrics.SWD import SWD
from gan.metrics.spectrum import Spectrum
from gan.metrics.bin_stats import BinStats

# Set path for the desired dataset:
dataset_path = os.path.join(data_path, 'clinical')

# Channel names of the dataset.
# This needs to be passed to the logging handler for better visualization.
channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

mapping = {'right': 0, 'rest': 1}

# Hyperparameter collection. 
# All Hyperparameters will be saved to wandb and can be seen on the project page.
GAN_PARAMS = {
    'n_channels':len(channels),
    'n_classes':2,
    'n_time':768,
    'n_stages':3,
    'n_filters':120,
    'fs':256,
    'latent_dim':210,
    'epochs_per_stage':50,
    'batch_size':128,
    'fading':True,
    }

# Init DataModule
dm = HDG(data_dir=dataset_path,
         n_stages=GAN_PARAMS['n_stages'],
         batch_size=GAN_PARAMS['batch_size'],
         num_workers=2)

# Init Logger
# Here we set the Project, Name and Folder for the run:
logger = WandbLogger(log_model=True,
                     name='reworked metric handler',
                     project='EEGGAN',
                     save_dir=results_path, )

# Init Checkpoint
# Here we set the rules for saving Model checkpoints:
checkpoint_callback = ModelCheckpoint(every_n_epochs=GAN_PARAMS['epochs_per_stage'],
                                    filename='checkpoint_{epoch}',
                                    save_top_k=-1,
                                    save_last=True,
                                    )

# Init logging handler
# Custom Callback for logging metrics and plots to wandb:
logging_handler = LoggingHandler()
logging_handler.attach_metrics([Spectrum(250),
                                SWD(1),
                                BinStats(channels, mapping, every_n_epochs = 0),
                                ],)

def main():
    model = GAN(**GAN_PARAMS)

    trainer = Trainer(
            max_epochs=GAN_PARAMS['epochs_per_stage'] * GAN_PARAMS['n_stages'],
            reload_dataloaders_every_n_epochs=GAN_PARAMS['epochs_per_stage'],
            callbacks=[Scheduler(), logging_handler, checkpoint_callback],
            default_root_dir=results_path,
            strategy='ddp_find_unused_parameters_true',
            logger=logger,
    )

    logger.watch(model)

    trainer.fit(model, dm)

if __name__ == '__main__':
    main()