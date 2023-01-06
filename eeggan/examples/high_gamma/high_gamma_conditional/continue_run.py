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
from eeggan.examples.high_gamma.models.conditional import Conditional
from eeggan.examples.high_gamma.train import train
from eeggan.model.builder import ProgressiveModelBuilder
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer

from torch.utils.tensorboard import SummaryWriter

n_epochs_per_stage = 5000

VERSION = 'continue_on_sub_stage3'
EXPERIMENT = 'ZCA_prewhitened'
DATASET = 'rest_right'

SUBJECT_ID = list(range(1,15))

RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'

STAGE = 3

writer = SummaryWriter(log_dir=f'/home/boehms/eeg-gan/EEG-GAN/Data/Tensorboard/{EXPERIMENT}/{VERSION}')

# Load state dics into generator and discriminator
STATE_DICT = torch.load(f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/ZCA_prewhitened/cGAN/1-14/states_stage_{STAGE-1}.pt')

DEFAULT_CONFIG = joblib.load('/home/boehms/eeg-gan/EEG-GAN/Data/Results/ZCA_prewhitened/cGAN/1-14/config.dict')
Model_Builder = joblib.load('/home/boehms/eeg-gan/EEG-GAN/Data/Results/ZCA_prewhitened/cGAN/1-14/model_builder.jblb')


def run(dataset_name: str, result_name: str, dataset_path: str, deep4_path: str, result_path: str,
        config: dict = DEFAULT_CONFIG, model_builder: ProgressiveModelBuilder = Model_Builder, stage: int = STAGE,
        state_dict = STATE_DICT):

    result_path_subj = os.path.join(result_path, result_name, str(dataset_name))
    
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
    trainer = GanSoftplusTrainer(500, discriminator, generator, config['r1_gamma'], config['r2_gamma'])

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, config['n_stages'], config['use_fade'],
                                             config['n_epochs_fade'], freeze_stages=config['freeze_stages'], current_stage=stage)
    
    progression_handler.set_progression(stage, 1.)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

    
    generator.train()
    discriminator.train()

    train(dataset_name=dataset_name,
        dataset_path=dataset_path,
        deep4s_path=deep4_path,
        result_path=result_path_subj,
        progression_handler=progression_handler,
        trainer=trainer,
        n_batch=config['n_batch'],
        lr_d=config['lr_d'],
        lr_g=config['lr_g'],
        betas=config['betas'],
        n_epochs_per_stage=n_epochs_per_stage,
        n_epochs_metrics=config['n_epochs_metrics'],
        plot_every_epoch=250,
        orig_fs=config['orig_fs'],
        n_samples=0,
        tensorboard_writer=writer
    )


if __name__ == "__main__":

    
    run(dataset_name=DATASET,
        result_name=VERSION,
        dataset_path=DATAPATH,
        deep4_path=MODELPATH,
        result_path=RESULTPATH,
        config=DEFAULT_CONFIG,
        model_builder=Model_Builder)
    
