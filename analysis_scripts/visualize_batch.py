# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import os 
import torch
from legacy_code.paths import results_path, data_path
import matplotlib.pyplot as plt
import numpy as np
from legacy_code.data.datamodule import HighGammaModule as HDG
from legacy_code.visualization.stft_plots import plot_bin_stats, calc_bin_stats
from legacy_code.visualization.spectrum_plots import plot_spectrum

fs = 128 
channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']


# Laod Dataset:
dataset_path = os.path.join(data_path, 'generated_stage4')
dm = HDG(dataset_path, 4, batch_size=128, num_workers=2)
dm.setup()

# draw 800 samples
idx = np.random.randint(0, len(dm.ds.data), 800)

fake = dm.ds.data[idx]
y_fake = dm.ds.target[idx]

fake = fake.detach().numpy()


MAPPING = {'right': 0, 'rest': 1}
conditional_dict = {}

for key in MAPPING.keys():
    conditional_dict[key] = [
        fake[y_fake == MAPPING[key]],
        fake[y_fake == MAPPING[key]]
        ]

# Dictionary with ffts for each condition on real and fake
resuts_dict = {}

out_path = '/home/boehms/eeg-gan/EEG-GAN/temp_plots/'

for key in MAPPING.keys():
    plot_bin_stats(conditional_dict[key][0], conditional_dict[key][1],
                    128, channels, out_path, str(key), False )
    

plt.close('all')

spectrum = plot_spectrum(fake, fake, fs, name='fake')

spectrum.savefig(os.path.join(out_path, 'spectrum_fake.png'), dpi=300)


