#  Author: Kay Hartmann <kg.hartma@gmail.com>
import os
from re import X
import joblib
from ignite.engine import Events
import sys

# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.examples.high_gamma.high_gamma_wgan.make_data_right_left import FS, N_PROGRESSIVE_STAGES, INPUT_LENGTH, MODELPATH, DATAPATH, EXPERIMENT
from eeggan.examples.high_gamma.models.baseline import Baseline
from eeggan.examples.high_gamma.train import train
from eeggan.model.builder import ProgressiveModelBuilder
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.wgan_gp import WganGpTrainer

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f'/home/boehms/eeg-gan/EEG-GAN/Data/Tensorboard/{EXPERIMENT}')

n_epochs_per_stage = 200

SUBJECT_ID = 1

RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'

DEFAULT_CONFIG = dict(
    n_chans=21,  # number of channels in data
    n_classes=2,  # number of classes in data
    orig_fs=FS,  # sampling rate of data

    n_batch=128,  # batch size
    n_stages=N_PROGRESSIVE_STAGES,  # number of progressive stages
    n_epochs_per_stage=n_epochs_per_stage,  # epochs in each progressive stage
    n_epochs_metrics=100,
    plot_every_epoch=100,
    n_epochs_fade=int(0.1 * n_epochs_per_stage),
    use_fade=False,
    freeze_stages=True,

    n_latent=200,  # latent vector size
    lambd=10,
    one_sided_penalty=False,
    distance_weighting=False,
    eps_drift=0.0001,
    eps_center=0,
    lr_d=0.005,  # discriminator learning rate
    lr_g=0.001,  # generator learning rate
    betas=(0., 0.99),  # optimizer betas

    n_filters=120,
    n_time=INPUT_LENGTH,

    upsampling='conv',
    downsampling='conv',
    discfading='cubic',
    genfading='cubic',
)

default_model_builder = Baseline(DEFAULT_CONFIG['n_stages'], DEFAULT_CONFIG['n_latent'], DEFAULT_CONFIG['n_time'],
                                 DEFAULT_CONFIG['n_chans'], DEFAULT_CONFIG['n_classes'], DEFAULT_CONFIG['n_filters'],
                                 upsampling=DEFAULT_CONFIG['upsampling'], downsampling=DEFAULT_CONFIG['downsampling'],
                                 discfading=DEFAULT_CONFIG['discfading'], genfading=DEFAULT_CONFIG['genfading'])


def run(subj_ind: int, result_name: str, dataset_path: str, deep4_path: str, result_path: str,
        summary_writer: SummaryWriter, 
        config: dict = DEFAULT_CONFIG, 
        model_builder: ProgressiveModelBuilder = default_model_builder
        ):



    result_path_subj = os.path.join(result_path, result_name, str(subj_ind))

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
    trainer = WganGpTrainer(10, discriminator, generator, config['lambd'], config['one_sided_penalty'],
    config['distance_weighting'], config['eps_drift'], config['eps_center'])

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, config['n_stages'], config['use_fade'],
                                             config['n_epochs_fade'], freeze_stages=config['freeze_stages'])
    progression_handler.set_progression(0, 1.)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    generator.train()
    discriminator.train()

    train(subj_ind, dataset_path, deep4_path, result_path_subj, progression_handler, trainer, config['n_batch'],
          config['lr_d'], config['lr_g'], config['betas'], config['n_epochs_per_stage'], config['n_epochs_metrics'],
          config['plot_every_epoch'], config['orig_fs'], tensorboard_writer=summary_writer)


if __name__ == "__main__":

    run(subj_ind=SUBJECT_ID,
        result_name='test',
        dataset_path=DATAPATH,
        deep4_path=MODELPATH,
        result_path=RESULTPATH,
        config=DEFAULT_CONFIG,
        model_builder=default_model_builder,
        summary_writer=writer)
    
