# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
import os 
import torch
from gan.paths import results_path, data_path
import sys

path = "EEGGAN/4vj37rcg"

model = GAN.load_from_checkpoint(
                checkpoint_path=os.path.join(results_path, path, "checkpoints/last.ckpt"),
                map_location=torch.device('cpu'),
                )

model.eval()
model.freeze()

# Set Stage:
model.generator.set_stage(5)
model.generator.fading = False
model.critic.set_stage(5)
model.critic.fading = False

print(model.current_epoch)

### test ###
from gan.data.datamodule import HighGammaModule as HDG
from gan.visualization.stft_plots import plot_bin_stats, calc_bin_stats
import numpy as np
from gan.visualization.spectrum_plots import plot_spectrum
import matplotlib.pyplot as plt

dataset_path = os.path.join(data_path, 'clinical')
dm = HDG(dataset_path, model.hparams.n_stages, batch_size=model.hparams.batch_size, num_workers=2)
dm.setup()

# draw 800 samples
idx = np.random.randint(0, len(dm.ds.data), 800)

real = dm.ds.data[idx]
y_real = dm.ds.target[idx]

real = model.critic.downsample(real, 0)

# generate noise and labels:
z = torch.randn(800, model.hparams.latent_dim)
z = z.type_as(real)

y_fake = torch.randint(low=0, high=model.hparams.n_classes,
                        size=(real.shape[0],), dtype=torch.int32)

# y_fake  = y_fake.type_as(y_real)

# generate fake batch:
fake = model.forward(z, y_fake)

fs = 128 
channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']


# stft_arr, f, t = compute_stft(real, fs, 0.2 * fs )

real = real.detach().numpy()
fake = fake.detach().numpy()

print(real.shape)
print(fake.shape)

# print('real shape: ', real.shape, 'fake shape: ', fake.shape)
MAPPING = {'right': 0, 'rest': 1}

conditional_dict = {}

for key in MAPPING.keys():
    conditional_dict[key] = [
        real[y_real == MAPPING[key]],
        fake[y_fake == MAPPING[key]]
        ]

# Dictionary with ffts for each condition on real and fake
resuts_dict = {}

out_path = '/home/boehms/eeg-gan/EEG-GAN/temp_plots/'

for key in MAPPING.keys():
    plot_bin_stats(conditional_dict[key][0], conditional_dict[key][1],
                    fs, channels, out_path, str(key), False )
    

plt.close('all')

spectrum = plot_spectrum(real, fake, fs, name='real vs fake')

spectrum.savefig(os.path.join(out_path, 'spectrum.png'), dpi=300)

