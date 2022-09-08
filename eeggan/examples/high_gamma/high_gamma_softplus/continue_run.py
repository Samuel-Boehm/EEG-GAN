#  Author: Kay Hartmann <kg.hartma@gmail.com>
import os
from re import X
import joblib
from ignite.engine import Events
import sys
import torch

# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.examples.high_gamma.high_gamma_softplus.make_data_rest_right import FS, N_PROGRESSIVE_STAGES, INPUT_LENGTH, MODELPATH, DATAPATH, EXPERIMENT
from eeggan.examples.high_gamma.models.baseline import Baseline
from eeggan.examples.high_gamma.train import train
from eeggan.model.builder import ProgressiveModelBuilder
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer

from torch.utils.tensorboard import SummaryWriter

n_epochs_per_stage = 20000

VERSION = 'v1'
EXPERIMENT = 'longrun_continue'

SUBJECT_ID = 1

RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'

STAGE = 3


writer = SummaryWriter(log_dir=f'/home/boehms/eeg-gan/EEG-GAN/Data/Tensorboard/{EXPERIMENT}/{VERSION}')

# Load state dics into generator and discriminator
STATE_DICT = torch.load(f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/longrun/v1/1/states_stage_{STAGE}.pt')


DEFAULT_CONFIG = dict(
    n_chans=21,  # number of channels in data
    n_classes=2,  # number of classes in data
    orig_fs=FS,  # sampling rate of data

    n_batch=128,  # batch size
    n_stages=N_PROGRESSIVE_STAGES,  # number of progressive stages
    n_epochs_per_stage=n_epochs_per_stage,  # epochs in each progressive stage
    n_epochs_metrics=50,
    plot_every_epoch=100,
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
)

default_model_builder = Baseline(DEFAULT_CONFIG['n_stages'], DEFAULT_CONFIG['n_latent'], DEFAULT_CONFIG['n_time'],
                                 DEFAULT_CONFIG['n_chans'], DEFAULT_CONFIG['n_classes'], DEFAULT_CONFIG['n_filters'],
                                 upsampling=DEFAULT_CONFIG['upsampling'], downsampling=DEFAULT_CONFIG['downsampling'],
                                 discfading=DEFAULT_CONFIG['discfading'], genfading=DEFAULT_CONFIG['genfading'])


def run(subj_ind: int, result_name: str, dataset_path: str, deep4_path: str, result_path: str,
        config: dict = DEFAULT_CONFIG, model_builder: ProgressiveModelBuilder = default_model_builder, stage: int = STAGE,
        state_dict = STATE_DICT):

    result_path_subj = os.path.join(result_path, result_name, str(subj_ind))
    
    os.makedirs(result_path_subj, exist_ok=True)

    joblib.dump(config, os.path.join(result_path_subj, 'config.dict'), compress=False)
    joblib.dump(model_builder, os.path.join(result_path_subj, 'model_builder.jblb'), compress=True)

    # create discriminator and generator modules
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()
    
    generator.cur_block = stage
    discriminator.cur_block = stage

    generator.load_state_dict(state_dict['generator'])
    discriminator.load_state_dict(state_dict['discriminator'])

    # trainer engine
    trainer = GanSoftplusTrainer(10, discriminator, generator, config['r1_gamma'], config['r2_gamma'])

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, config['n_stages'], config['use_fade'],
                                             config['n_epochs_fade'], freeze_stages=config['freeze_stages'])
    
    progression_handler.set_progression(stage, 1.)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    print(progression_handler.generator.cur_block)
    print(progression_handler.discriminator.cur_block)

    generator.train()
    discriminator.train()

    train(subj_ind, dataset_path, deep4_path, result_path_subj, progression_handler, trainer, config['n_batch'],
          config['lr_d'], config['lr_g'], config['betas'], config['n_epochs_per_stage'], config['n_epochs_metrics'],
          config['plot_every_epoch'], config['orig_fs'], writer)


if __name__ == "__main__":

    
    run(subj_ind=SUBJECT_ID,
        result_name=VERSION,
        dataset_path=DATAPATH,
        deep4_path=MODELPATH,
        result_path=RESULTPATH,
        config=DEFAULT_CONFIG,
        model_builder=default_model_builder)
    
