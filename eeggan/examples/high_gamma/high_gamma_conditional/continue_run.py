#  Author: Samuel Böhm <samuel-boehm@web.de>
import os
from re import X
import joblib
from ignite.engine import Events
import sys
import argparse

# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.examples.high_gamma.train import train
from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from eeggan.examples.high_gamma.load_gan import load_GAN
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, required=True)
args = parser.parse_args()

n_epochs_per_stage = 2000

VERSION = 'single_subjects_on_pretrained'
EXPERIMENT = 'ZCA_prewhitened'
DATASET = 'rest_right'

DATAPATH = '/home/boehms/eeg-gan/EEG-GAN/Data/Data/ZCA_prewhitened'
DEEP4_PATH = '/home/boehms/eeg-gan/EEG-GAN/Data/Models/ZCA_prewhitened'
RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'
STAGE = 0
SUBJECT = args.subject
GAN_PATH = '/home/boehms/eeg-gan/EEG-GAN/Data/Results/ZCA_prewhitened/cGAN/1-14/'
DEFAULT_CONFIG = joblib.load('/home/boehms/eeg-gan/EEG-GAN/Data/Results/ZCA_prewhitened/cGAN/1-14/config.dict')

writer = SummaryWriter(log_dir=f'/home/boehms/eeg-gan/EEG-GAN/Data/Tensorboard/{EXPERIMENT}/{VERSION}')



def run(dataset_name: str, result_name: str, dataset_path: str, deep4_path: str, result_path: str,
        config: dict = DEFAULT_CONFIG,  stage: int = STAGE,
        GAN_path = GAN_PATH, subject: int = None):

    result_path_subj = os.path.join(result_path, result_name, dataset_name, str(subject))
    
    os.makedirs(result_path_subj, exist_ok=True)

    joblib.dump(config, os.path.join(result_path_subj, 'config.dict'), compress=False)

    generator, discriminator, model_builder = load_GAN(GAN_path, stage)

    joblib.dump(model_builder, os.path.join(result_path_subj, 'model_builder.jblb'), compress=True)

    # trainer engine
    trainer = GanSoftplusTrainer(500, discriminator, generator, config['r1_gamma'], config['r2_gamma'])

    # handles potential progression after each epoch
    progression_handler = ProgressionHandler(discriminator, generator, config['n_stages'], config['use_fade'],
                                             config['n_epochs_fade'], freeze_stages=config['freeze_stages'], current_stage=stage)
    
    progression_handler.set_progression(stage, 1.)
    
    #Set max stage to stage. So we are only training this stage and not stages afterwards
    # progression_handler.n_stages = stage + 1
    
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
        plot_every_epoch=1000,
        orig_fs=config['orig_fs'],
        n_samples=0,
        tensorboard_writer=writer,
        subject=subject,
        pretrained_path=GAN_path
    )


if __name__ == "__main__":

    
    run(dataset_name=DATASET,
        result_name=VERSION,
        dataset_path=DATAPATH,
        deep4_path=DEEP4_PATH,
        result_path=RESULTPATH,
        config=DEFAULT_CONFIG,
        GAN_path = GAN_PATH,
        subject=SUBJECT)
    
