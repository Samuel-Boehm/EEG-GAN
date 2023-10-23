# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
import os 
import torch
from gan.paths import results_path, data_path
from gan.data.DataModule import HighGammaModule as HDG
from gan.handler.ProgressionHandler import Scheduler
from gan.handler.LoggingHandler import LoggingHandler
from gan.paths import data_path, results_path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
import numpy as np

# Import metrics:
from gan.metrics.SWD import SWD
from gan.metrics.spectrum import Spectrum
from gan.metrics.bin_stats import BinStats


path = "EEGGAN/twlqedvh"
dataset_path = os.path.join(data_path, 'clinical')

dataset = torch.load(dataset_path)

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

mapping = {'right': 0, 'rest': 1}

GAN_PARAMS = {
    'n_channels':len(channels),
    'n_classes':2,
    'n_time':768,
    'n_stages':5,
    'n_filters':120,
    'fs':256,
    'latent_dim':210,
    'epochs_per_stage': [200, 400, 600, 600, 600], 
    'batch_size':128,
    'fading':True,
    'alpha':1,
    'beta':.01,
    'freeze':True,
    }



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
training_schedule = Scheduler(fading_period=50)

# Init DataModule
dm = HDG(dataset_path, GAN_PARAMS['n_stages'], batch_size=GAN_PARAMS['batch_size'], num_workers=2)
dm.setup()

# Init Logger


def main():

    for i in range(2, 15):
        model = GAN.load_from_checkpoint(
                checkpoint_path=os.path.join(results_path, path, "checkpoints/last.ckpt"),
                map_location=torch.device('cpu'),
                )

        # Set Stage:
        model.generator.set_stage(1)
        model.critic.set_stage(1)

        model.generator.fading = False
        model.critic.fading = False
        
        train_idx, test_idx = dataset.splits['idx'][dataset.splits.subject == i].values

        # For GAN training we pool all data together:
        idx = np.concatenate((train_idx, test_idx))

        dm.select_subset(idx)

        # Init Logger and Trainer for each subject:

        logger = WandbLogger(name=f'Subject_training{i}', project='EEGGAN', save_dir=results_path,)

        trainer = Trainer(
            max_epochs=int(np.sum(GAN_PARAMS['epochs_per_stage'])),
            reload_dataloaders_every_n_epochs=1,
            callbacks=[training_schedule, logging_handler, checkpoint_callback],
            default_root_dir=results_path,
            strategy='ddp_find_unused_parameters_true',
            logger=logger,)

        logger.watch(model)

        trainer.fit(model, dm)

if __name__ == '__main__':
    main()