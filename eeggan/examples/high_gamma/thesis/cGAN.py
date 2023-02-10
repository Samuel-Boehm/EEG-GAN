#  Author: Kay Hartmann <kg.hartma@gmail.com>
import os
from re import X
import joblib
from ignite.engine import Events
import sys

# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.examples.high_gamma.models.conditional import Conditional
from eeggan.examples.high_gamma.train import train
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from torch.utils.tensorboard import SummaryWriter

FS = 512.
SEGMENT_IVAL = (-0.5, 4.00)
INPUT_LENGTH = int((SEGMENT_IVAL[1] - SEGMENT_IVAL[0]) * FS)
N_PROGRESSIVE_STAGES = 6
N_EPOCHS_PER_STAGE = 1000

EXPERIMENT = 'Thesis'
DATASET = 'whitened'
VERSION = 'cGAN'

DATAPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/{EXPERIMENT}'
DEEP4PATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Models/{EXPERIMENT}'
RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'

writer = SummaryWriter(log_dir=f'/home/boehms/eeg-gan/EEG-GAN/Data/Tensorboard/{EXPERIMENT}/{VERSION}')


config = dict(
    n_chans=21,  # number of channels in data
    n_classes=2,  # number of classes in data
    orig_fs=FS,  # sampling rate of data
    n_batch=128,  # batch size
    n_stages=N_PROGRESSIVE_STAGES,  # number of progressive stages
    n_epochs_per_stage=N_EPOCHS_PER_STAGE,  # epochs in each progressive stage
    n_epochs_metrics=200,
    plot_every_epoch=500,
    n_epochs_fade=int(0.1 * N_EPOCHS_PER_STAGE),
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
    n_samples='all'
)
model_builder = Conditional(config['n_stages'],
                    config['n_latent'],
                    config['n_time'],
                    config['n_chans'],
                    config['n_classes'],
                    config['n_filters'],
                    upsampling=config['upsampling'],
                    downsampling=config['downsampling'],
                    discfading=config['discfading'],
                    genfading=config['genfading'])

result_path_subj = os.path.join(RESULTPATH, EXPERIMENT, DATASET)

os.makedirs(result_path_subj, exist_ok=True)

joblib.dump(config, os.path.join(result_path_subj, 'config.dict'), compress=False)
joblib.dump(model_builder, os.path.join(result_path_subj, 'model_builder.jblb'), compress=True)

generator = model_builder.build_generator()
discriminator = model_builder.build_discriminator()

generator.apply(weight_filler)
discriminator.apply(weight_filler)

# trainer engine
trainer = GanSoftplusTrainer(i_logging=200,
                        discriminator=discriminator,
                        generator=generator,
                        r1_gamma=config['r1_gamma'],
                        r2_gamma=config['r2_gamma'])

# handles potential progression after each epoch
progression_handler = ProgressionHandler(discriminator, 
                        generator,
                        config['n_stages'],
                        config['use_fade'],
                        config['n_epochs_fade'],
                        freeze_stages=config['freeze_stages'])
                                            
progression_handler.set_progression(0, 1.)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

generator.train()
discriminator.train()

if __name__ == '__main__':
    train(DATASET, DATAPATH, DEEP4PATH, result_path_subj, progression_handler,
    trainer, config['n_batch'], config['lr_d'], config['lr_g'], config['betas'],
    config['n_epochs_per_stage'], config['n_epochs_metrics'], config['plot_every_epoch'],
    config['orig_fs'], config['n_samples'], writer, None, None)