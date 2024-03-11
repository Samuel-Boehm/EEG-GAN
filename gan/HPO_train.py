# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
from lightning import Trainer
import numpy as np
import os
import wandb
from gan.data.datamodule import HighGammaModule as HDG
from gan.handler.progressionhandler import Scheduler
from gan.handler.logginghandler import LoggingHandler
from gan.paths import data_path, results_path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import torch


# Import metrics:
from gan.metrics.SWD import SWD
from gan.metrics.spectrum import Spectrum
from gan.metrics.bin_stats import BinStats

# Import metadata:
from constants import CHANNELS, MAPPING

# Define dataset to use
dataset_path = os.path.join(data_path, 'clinical')

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
    'epochs_per_stage': [200, 300, 400, 500, 500], 
    'batch_size':128,
    'fading':True,
    'alpha':1,
    'beta':.1,
    'freeze':True,
    }

def parse_args():
    '''Overwrite default parameters with command line arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_channels', type=int, default=GAN_PARAMS['n_channels'], 
                        help='Number of channels in the recorded EEG data')
    parser.add_argument('--n_classes', type=int, default=GAN_PARAMS['n_classes'],
                        help='Number of classes/labels in the dataset')
    parser.add_argument('--n_time', type=int, default=GAN_PARAMS['n_time'],
                        help='Number of time steps in the recorded EEG data')
    parser.add_argument('--n_stages', type=int, default=GAN_PARAMS['n_stages'],
                        help='Number of stages in the training process')
    parser.add_argument('--n_filters', type=int, default=GAN_PARAMS['n_filters'],
                        help='Number of filters in the convolutional layers')
    parser.add_argument('--fs', type=int, default=GAN_PARAMS['fs'],
                        help='Sampling frequency of the recorded EEG data')
    parser.add_argument('--latent_dim', type=int, default=GAN_PARAMS['latent_dim'],
                        help='Dimension of the latent space for conditioning the generator')
    parser.add_argument('--epochs_per_stage', type=list, default=GAN_PARAMS['epochs_per_stage'],
                        help='Number of epochs to train per stage, length must be equal to n_stages')
    parser.add_argument('--batch_size', type=int, default=GAN_PARAMS['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--fading', type=bool, default=GAN_PARAMS['fading'],
                        help='Whether to use fading between stages in the training process')
    parser.add_argument('--alpha', type=float, default=GAN_PARAMS['alpha'],
                        help='Alpha parameter for the spectral loss, this regulates the ammount of\
                            the time domain loss in the spectral loss')
    parser.add_argument('--beta', type=float, default=GAN_PARAMS['beta'],
                        help='Beta parameter for the spectral loss, this regulates the ammount of\
                            the frequency domain loss in the spectral loss')
    parser.add_argument('--freeze', type=bool, default=GAN_PARAMS['freeze'],
                        help='Whether to freeze already trained stages in the training process')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--wandb_resume_version', type=str,)
    parser.add_argument('--devices', type=int, default=4 )
    parser.add_argument('--num_nodes', type=int, default=2 )
    args = parser.parse_args()

    # Create a new dictionary that only contains the keys that are in both args and GAN_PARAMS
    args_dict = {k: v for k, v in vars(args).items() if k in GAN_PARAMS}

    # Update GAN_PARAMS with this new dictionary
    GAN_PARAMS.update(args_dict)
    return args


def main():

    print('Number of GPUs seen:', torch.cuda.device_count())
    # print('Resume version: ', args.wandb_resume_version)

    args = parse_args()

    # Init DataModule
    dm = HDG(dataset_path, GAN_PARAMS['n_stages'], batch_size=GAN_PARAMS['batch_size'],
            num_workers=2)

    # Init Logger
    logger = WandbLogger(name=args.experiment_name, version=args.wandb_resume_version, resume="allow")

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

    
    model = GAN(**GAN_PARAMS)

    trainer = Trainer(
            devices=args.devices, num_nodes=args.num_nodes, 
            accelerator='gpu',
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