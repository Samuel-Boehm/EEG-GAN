
import os
import joblib
import sys
from ignite.engine import Events

# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.examples.high_gamma.models.spconGAN import SP_GAN
from eeggan.training.progressive.handler import SpectralProgessionHandler
from eeggan.training.trainer.gan_softplus_spectral import SpectralTrainer
from eeggan.pytorch.utils.weights import weight_filler
from eeggan.examples.high_gamma.train_spectral import train_spectral
from eeggan.examples.high_gamma.high_gamma_softplus.make_data_rest_right import (
FS, N_PROGRESSIVE_STAGES, INPUT_LENGTH
)

n_epochs_per_stage = 500
EXPERIMENT = 'ZCA_prewhitened'
VERSION = 'SP_CONV_GAN'
DATASET = 'rest_right'
RESULTPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Results/{EXPERIMENT}'
DEEP4_PATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Models/ZCA_prewhitened'
DATAPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/ZCA_prewhitened'

config = dict(
    n_chans=21,  # number of channels in data
    n_classes=2,  # number of classes in data
    orig_fs=FS,  # sampling rate of data
    n_batch=64,  # batch size
    n_stages=N_PROGRESSIVE_STAGES,  # number of progressive stages
    n_epochs_per_stage=n_epochs_per_stage,  # epochs in each progressive stage
    n_epochs_metrics=50,
    plot_every_epoch=250,
    n_epochs_fade=int(0.1 * n_epochs_per_stage),
    use_fade=False,
    freeze_stages=True,
    n_latent=200,  # latent vector size
    r1_gamma=10., # lambda/gamma for gradient penalty
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
    n_samples=6742
)

model_builder = SP_GAN(config['n_stages'],
                    config['n_latent'],
                    config['n_time'],
                    config['n_chans'],
                    config['n_classes'],
                    config['n_filters'],
                    upsampling=config['upsampling'],
                    downsampling=config['downsampling'],
                    discfading=config['discfading'],
                    genfading=config['genfading'])

result_path_subj = os.path.join(RESULTPATH, VERSION, DATASET)

os.makedirs(result_path_subj, exist_ok=True)

joblib.dump(config, os.path.join(result_path_subj, 'config.dict'), compress=False)
joblib.dump(model_builder, os.path.join(result_path_subj, 'model_builder.jblb'), compress=True)


generator = model_builder.build_generator()
discriminator = model_builder.build_discriminator()
spectral_discriminator = model_builder.build_spectral_discriminator()

generator.apply(weight_filler)
discriminator.apply(weight_filler)
spectral_discriminator.apply(weight_filler)

# trainer engine
trainer = SpectralTrainer(i_logging=10,
                        discriminator=discriminator,
                        generator=generator,
                        r1=config['r1_gamma'],
                        spectral_discriminator=spectral_discriminator)

# handles potential progression after each epoch
progression_handler = SpectralProgessionHandler(discriminator, 
                        generator, 
                        spectral_discriminator,
                        config['n_stages'],
                        config['use_fade'],
                        config['n_epochs_fade'],
                        freeze_stages=config['freeze_stages'])
                                            
progression_handler.set_progression(0, 1.)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), progression_handler.advance_alpha)

generator.train();
discriminator.train();
spectral_discriminator.train();

if __name__ == "__main__":

    train_spectral(DATASET, DATAPATH, DEEP4_PATH, result_path_subj, progression_handler, trainer, config['n_batch'],
        config['lr_d'], config['lr_g'], config['betas'], config['n_epochs_per_stage'], config['n_epochs_metrics'],
        config['plot_every_epoch'], config['orig_fs'], config['n_samples'], None, 4, None)