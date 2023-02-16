
import os
import joblib
from ignite.engine import Events
import sys
from torch.utils.tensorboard import SummaryWriter


# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.training.progressive.handler import ProgressionHandler
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from eeggan.examples.high_gamma.train import train
from eeggan.examples.high_gamma.load_gan import load_GAN

EXPERIMENT = 'Thesis'
DATASET = 'baseline'
VERSION = 'baseline'

DATAPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/{EXPERIMENT}'
DEEP4PATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Models/{EXPERIMENT}'
RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'


CHECKPOINTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}/{VERSION}/'
STAGE = 3

config = joblib.load(os.path.join( CHECKPOINTPATH, 'config.dict'))

generator, discriminator, model_builder = load_GAN(CHECKPOINTPATH, STAGE)

result_path_subj = os.path.join(RESULTPATH, VERSION)

os.makedirs(result_path_subj, exist_ok=True)

joblib.dump(config, os.path.join(result_path_subj, 'config.dict'), compress=False)
joblib.dump(model_builder, os.path.join(result_path_subj, 'model_builder.jblb'), compress=True)

# trainer engine
trainer = GanSoftplusTrainer(i_logging=10,
                        discriminator=discriminator,
                        generator=generator,
                        r1=config['r1_gamma'],
                        r2=config['r2_gamma'],
                        )

# handles potential progression after each epoch
progression_handler = ProgressionHandler(discriminator, 
                        generator,
                        config['n_stages'],
                        config['use_fade'],
                        config['n_epochs_fade'],
                        freeze_stages=config['freeze_stages'])
                                            
progression_handler.set_progression(STAGE+1, 1.)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

generator.train();
discriminator.train();

if __name__ == '__main__':
    train(DATASET, DATAPATH, DEEP4PATH, result_path_subj, progression_handler,
    trainer, config['n_batch'], config['lr_d'], config['lr_g'], config['betas'],
    config['n_epochs_per_stage'], config['n_epochs_metrics'], config['plot_every_epoch'],
    config['orig_fs'], config['n_samples'], None, None, None)