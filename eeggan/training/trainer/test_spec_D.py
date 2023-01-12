# Temporal file to test in implement the spectral discriminator

import sys

# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')


from eeggan.training.spectral_discriminator import SpectralDiscriminator
from eeggan.examples.high_gamma.make_data import load_dataset


batch_real = load_dataset('rest_right', '/home/boehms/eeg-gan/EEG-GAN/Data/Data/ZCA_prewhitened')

print(batch_real)