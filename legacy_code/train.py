# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from lightning import Trainer
import numpy as np
from pathlib import Path

from legacy_code.model.gan import GAN
from legacy_code.data.datamodule import ProgressiveGrowingDataset
from legacy_code.handler.progression import Scheduler
from legacy_code.handler.logging import LoggingHandler

from lightning.pytorch.loggers import WandbLogger

import hydra
from omegaconf import DictConfig

# Import metrics:
from metrics.SWD import SWD
from legacy_code.metrics.spectrum import Spectrum
from legacy_code.metrics.bin_stats import BinStats


channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

mapping = {'right': 0, 'rest': 1}

# Collect all parameters for the training here:
GAN_PARAMS = {
    'n_channels':len(channels),
    'n_classes':2,
    'n_time':768, # care with this weird hardcoding, its fs [Hz] * time [s]
    'n_stages':5,
    'n_filters':120,
    'fs':256,
    'latent_dim':210,
    'epochs_per_stage': [1000, 1000, 1000, 1000, 1000], 
    'batch_size':32,
    'fading':True,
    'alpha':1,
    'beta':0.01,
    'freeze':True,
    'lr_critic':0.005,
    'lr_gen':0.001,
    'n_critic':1,
    }


def train(cfg: DictConfig) -> None:

    base_dir = Path.cwd().parent
    data_path = Path(base_dir, 'datasets', cfg.dataset.dataset_name)
    dm = ProgressiveGrowingDataset(data_path, cfg.model.n_stages, cfg.training.batch_size)

    # Init Logger
    logger = WandbLogger(name='GAN', project='EEGGAN', save_dir=cfg.run.dir)

    # Init logging handler
    logging_handler = LoggingHandler()
    logging_handler.attach_metrics([SWD(1)])

    # Init Scheduler
    training_schedule = Scheduler(fading_period=50)



    model = GAN(**cfg.model)

    trainer = Trainer(**cfg.trainer,
            # limit_train_batches=1, # batches are limited for debugging
            max_epochs=int(np.sum(GAN_PARAMS['epochs_per_stage'])),
            reload_dataloaders_every_n_epochs=1,
            callbacks=[training_schedule, logging_handler, checkpoint_callback],
            default_root_dir=results_path,
            strategy='ddp_find_unused_parameters_true',
            logger=logger,
    )

    logger.watch(model)

    trainer.fit(model, dm)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    train()

if __name__ == '__main__':
    main()