# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
from lightning import Trainer
import numpy as np
import os
from gan.data.DataModule import HighGammaModule as HDG
from gan.handler.ProgressionHandler import Scheduler
from gan.handler.LoggingHandler import LoggingHandler
from gan.paths import data_path, results_path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Import metrics:
from gan.metrics.SWD import SWD
from gan.metrics.spectrum import Spectrum
from gan.metrics.bin_stats import BinStats

# Define dataset to use
dataset_path = os.path.join(data_path, 'clinical')

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

mapping = {'right': 0, 'rest': 1}

# Collect all parameters for the training here:
GAN_PARAMS = {
    'n_channels':len(channels),
    'n_classes':2,
    'n_time':768,
    'n_stages':5,
    'n_filters':120,
    'fs':256,
    'latent_dim':210,
    'epochs_per_stage': [200, 400, 800, 1600, 2000], 
    'batch_size':128,
    'fading':True,
    'alpha':1,
    'beta':.01,
    }

# Init DataModule
dm = HDG(dataset_path, GAN_PARAMS['n_stages'], batch_size=GAN_PARAMS['batch_size'], num_workers=2)

# Init Logger
logger = WandbLogger(name='weight_norm', project='EEGGAN', save_dir=results_path,)

# Init Checkpoint
checkpoint_callback = ModelCheckpoint(every_n_epochs=500,
                                    filename='checkpoint_{epoch}',
                                    save_last=True,
                                    )

# Init logging handler
logging_handler = LoggingHandler()
logging_handler.attach_metrics([Spectrum(100),
                                SWD(1),
                                BinStats(channels, mapping, every_n_epochs=0),
                                ],)

# Init Scheduler
training_schedule = Scheduler(fading_period=100)

def main():
    model = GAN(**GAN_PARAMS)

    trainer = Trainer(
            max_epochs=int(np.sum(GAN_PARAMS['epochs_per_stage'])),
            reload_dataloaders_every_n_epochs=1,
            callbacks=[training_schedule, logging_handler, checkpoint_callback],
            default_root_dir=results_path,
            strategy='ddp_find_unused_parameters_true',
            logger=logger,
    )

    logger.watch(model)

    trainer.fit(model, dm)

if __name__ == '__main__':
    main()