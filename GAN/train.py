# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from Model.GAN import GAN
from lightning import Trainer
import torch
import os
from Data.DataModule import HighGammaModule as HDG
from Handler.ProgressionHandler import Scheduler
from Handler.LoggingHandler import LoggingHandler
from paths import data_path, results_path
from lightning.pytorch.loggers import WandbLogger



# Define dataset to use
dataset_path = os.path.join(data_path, 'clinical')

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

# Collect all parameters for the training here:
GAN_PARAMS = {
    'n_channels':len(channels),
    'n_classes':2,
    'n_time':768,
    'n_stages':6,
    'n_filters':120,
    'fs':256,
    'latent_dim':210,
    'epochs_per_stage':1000,
    }

# Init DataModule
dm = HDG(dataset_path, GAN_PARAMS['n_stages'], batch_size=32, num_workers=2)

# Init Logger
logger = WandbLogger(name='baseline', project='EEGGAN', save_dir=results_path)

def main():
    model = GAN(**GAN_PARAMS)

    trainer = Trainer(
            max_epochs=GAN_PARAMS['epochs_per_stage'] * GAN_PARAMS['n_stages'],
            reload_dataloaders_every_n_epochs=GAN_PARAMS['epochs_per_stage'],
            callbacks=[Scheduler(), LoggingHandler()],
            default_root_dir=results_path,
            strategy='ddp_find_unused_parameters_true',
            logger=logger,
    )

    logger.watch(model)


    # Create folder for plots:
    # print(trainer.logger.log_dir)
    #plots_path = os.path.join(trainer.logger.log_dir, 'plots')
    #if not os.path.exists(plots_path):
    #    os.makedirs(plots_path)

    trainer.fit(model, dm)

if __name__ == '__main__':
    main()