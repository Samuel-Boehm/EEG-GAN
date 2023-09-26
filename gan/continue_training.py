# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
from lightning import Trainer
import torch
import os
import wandb
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

### set checkpoint path ###
model_hash = "EEGGAN/20k67cbw" # Hash from w and b
checkpoint_path=os.path.join(results_path, model_hash, "checkpoints/last.ckpt")

# Set path for the desired dataset:
dataset_path = os.path.join(data_path, 'clinical')

# Load config and h_params from w and b API
api = wandb.Api()
run = api.run(model_hash)
config = run.config


# Set some parameters for the metrics
mapping = {'right': 0, 'rest': 1}

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

# Init DataModule
dm = HDG(data_dir=dataset_path,
         n_stages=config.n_stages,
         batch_size=config.batch_size,
         num_workers=2)

# Init Logger
# Here we set the Project, Name and Folder for the run:
logger = WandbLogger(log_model="all",
                     name='try resuming',
                     project='EEGGAN',
                     save_dir=results_path, )

# Init Checkpoint
# Here we set the rules for saving Model checkpoints:
checkpoint_callback = ModelCheckpoint(every_n_epochs=config.epochs_per_stage,
                                    filename='checkpoint_{epoch}',
                                    save_top_k=-1,
                                    save_last=True,
                                    )

# Init logging handler
# Custom Callback for logging metrics and plots to wandb:
logging_handler = LoggingHandler()
logging_handler.attach_metrics([Spectrum(1),
                                SWD(1),
                                BinStats(channels, mapping, every_n_epochs = 0),
                                ],)

def main():
    model = GAN(**config)

    trainer = Trainer(
            max_epochs=config.epochs_per_stage * config.n_stages,
            reload_dataloaders_every_n_epochs=config.epochs_per_stage,
            callbacks=[Scheduler(), logging_handler, checkpoint_callback],
            default_root_dir=results_path,
            strategy='ddp_find_unused_parameters_true',
            logger=logger,
    )

    logger.watch(model)

    trainer.fit(model, dm, ckpt_path=checkpoint_path,)

if __name__ == '__main__':
    main()