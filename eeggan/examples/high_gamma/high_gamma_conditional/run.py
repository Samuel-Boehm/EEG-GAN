#  Author: Kay Hartmann <kg.hartma@gmail.com>
import os
from re import X
import joblib
from ignite.engine import Events
import sys
import argparse

# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.examples.high_gamma.high_gamma_softplus.make_data_rest_right import FS, N_PROGRESSIVE_STAGES, MODELPATH, DATAPATH
from eeggan.examples.high_gamma.models.conditionalV2 import ConditionalV2
from eeggan.examples.high_gamma.train import train
from eeggan.model.builder import ProgressiveModelBuilder
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from torch.utils.tensorboard import SummaryWriter

n_epochs_per_stage = 1000
EXPERIMENT = 'ZCA_prewhitened'
VERSION = 'cGAN_short'

SEGMENT_IVAL = (-0.5, 2.00)
INPUT_LENGTH = int((SEGMENT_IVAL[1] - SEGMENT_IVAL[0]) * FS)


DATASET = 'rest_right_2'

RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'

writer = SummaryWriter(log_dir=f'/home/boehms/eeg-gan/EEG-GAN/Data/Tensorboard/{EXPERIMENT}/{VERSION}')


DEFAULT_CONFIG = dict(
    n_chans=21,  # number of channels in data
    n_classes=2,  # number of classes in data
    orig_fs=FS,  # sampling rate of data

    n_batch=128,  # batch size
    n_stages=N_PROGRESSIVE_STAGES,  # number of progressive stages
    n_epochs_per_stage=n_epochs_per_stage,  # epochs in each progressive stage
    n_epochs_metrics=50,
    plot_every_epoch=250,
    n_epochs_fade=int(0.1 * n_epochs_per_stage),
    use_fade=False,
    freeze_stages=True,

    n_latent=200,  # latent vector size
    r1_gamma=10.,
    r2_gamma=0.,
    lr_d=0.005,  # discriminator learning rate
    lr_g=0.001,  # generator learning rate
    betas=(0., 0.99),  # optimizer betas

    n_filters=120,
    n_time=INPUT_LENGTH,

    upsampling='conv',
    downsampling='conv',
    discfading='cubic',
    genfading='cubic',
    n_samples=6742  #int(2240 + 11244)
)

default_model_builder = ConditionalV2(DEFAULT_CONFIG['n_stages'], DEFAULT_CONFIG['n_latent'], DEFAULT_CONFIG['n_time'],
                                 DEFAULT_CONFIG['n_chans'], DEFAULT_CONFIG['n_classes'], DEFAULT_CONFIG['n_filters'],
                                 upsampling=DEFAULT_CONFIG['upsampling'], downsampling=DEFAULT_CONFIG['downsampling'],
                                 discfading=DEFAULT_CONFIG['discfading'], genfading=DEFAULT_CONFIG['genfading'])


def run(dataset_name: str, result_name: str, dataset_path: str, deep4_path: str, result_path: str,
        config: dict = DEFAULT_CONFIG, model_builder: ProgressiveModelBuilder = default_model_builder):

    result_path_subj = os.path.join(result_path, result_name, dataset_name)
    
    os.makedirs(result_path_subj, exist_ok=True)

    joblib.dump(config, os.path.join(result_path_subj, 'config.dict'), compress=False)
    joblib.dump(model_builder, os.path.join(result_path_subj, 'model_builder.jblb'), compress=True)

    # create discriminator and generator modules
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()

    # initiate weights
    generator.apply(weight_filler) # Apply is part of nn.module and applies a function over a whole network
    discriminator.apply(weight_filler)

    # trainer engine
    trainer = GanSoftplusTrainer(10, discriminator, generator, config['r1_gamma'], config['r2_gamma'])

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, config['n_stages'], config['use_fade'],
                                             config['n_epochs_fade'], freeze_stages=config['freeze_stages'])
    progression_handler.set_progression(0, 1.)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    generator.train()
    discriminator.train()

    train(dataset_name, dataset_path, deep4_path, result_path_subj, progression_handler, trainer, config['n_batch'],
          config['lr_d'], config['lr_g'], config['betas'], config['n_epochs_per_stage'], config['n_epochs_metrics'],
          config['plot_every_epoch'], config['orig_fs'], config['n_samples'], writer)


if __name__ == "__main__":

    
    run(dataset_name=DATASET,
        result_name=VERSION,
        dataset_path=DATAPATH,
        deep4_path=MODELPATH,
        result_path=RESULTPATH,
        config=DEFAULT_CONFIG,
        model_builder=default_model_builder)
    
