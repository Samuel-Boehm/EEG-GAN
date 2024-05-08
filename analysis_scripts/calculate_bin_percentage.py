from legacy_code.visualization.stft_plots import plot_stft, calc_bin_stats, calc_bin_stats_for_mapping, plot_bin_stats
from legacy_code.visualization.spectrum_plots import plot_spectrum
from legacy_code.data.batch import batch_data
from legacy_code.paths import data_path
import numpy as np
from legacy_code.utils import generate_data
from legacy_code.data.utils import downsample_step
import matplotlib.pyplot as plt

import torch
import os

subject = 4

fs = 128

mapping = {'right': 0, 'rest': 1}

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

# Load Data
real = torch.load(os.path.join(data_path, 'clinical'))
real.select_from_tag('subject', subject)

real.data = downsample_step(real.data)

run_id = "ui630z5a"
stage = 4
fake = generate_data(run_id, stage, 1000)

# Subsample the smaller dataset:
if fake.data.shape[0] > real.data.shape[0]:
    idx = np.random.choice(fake.data.shape[0], real.data.shape[0], replace=False)
    fake.data = fake.data[idx]
    fake.target =  fake.target[idx]

elif fake.data.shape[0] < real.data.shape[0]:
    idx = np.random.choice(real.data.shape[0], fake.data.shape[0], replace=False)
    real.data = real.data[idx]
    real.target =  real.target[idx]
else: 
    pass


# Metrics work best with batched data:
batch = batch_data(real.data, fake.data, real.target, fake.target)

print(f'batch real dimensions: {batch.real.shape}')
print(f'batch fake dimensions: {batch.fake.shape}')


stats_dict = calc_bin_stats_for_mapping(batch.real, batch.y_real, batch.fake, batch.y_fake,
                                        fs, mapping)

for key in mapping.keys():
        _, _, p_vals, _, _  = stats_dict[key]
        
        different = np.sum(p_vals < 0.3)
        # pvals shape should be batch x channels x bin time x bin freq
        for i, name in enumerate(channels):
             no_difference = np.sum(p_vals[i, : , : ] >= 0.3)
             print(f'{key} {name}: % no difference {(no_difference/p_vals[i, : , : ].size):.2f}')

# Plot single trials: 

# We want to plot the spectrum and the STFT of 100 trials of each class and save them as .png files.


def single_stft(real_X, fake_X, title, channels, fs, path=None):

    real_stft, fake_stft, _, f, t = calc_bin_stats(real_X, fake_X, fs)

    if path:
        path_real=os.path.join(path, f'{title}_real_{fs}.png')
        path_fake=os.path.join(path, f'{title}_fake_{fs}.png')
    else:
        path_real = None
        path_fake = None
                          
    plot_stft(
            real_stft, t, f, f'{title} {fs} Hz - p value > 0.3', 'log10 rel. power mean', 
            channel_names=channels, upper=5, lower = -5, path=path_real,
            )
    
    plot_stft(
            fake_stft, t, f, f'{title} {fs} Hz - p value > 0.3', 'log10 rel. power mean',
            channel_names=channels, upper=5, lower = -5, path=path_fake,
            )

def plot_single_trials(real_X, real_y, fake_X, fake_y, mapping, fs, channels, n_trials, path):
    """
    Plot the spectrum and the STFT of n_trials trials of each class and save them as .png files.
    """
    for key in mapping.keys():
        # print(np.where(real_y == mapping[key]), real_y, mapping[key])
        real_idx = np.where(real_y == mapping[key])[0]
        fake_idx = np.where(fake_y == mapping[key])[0]
        real_idx = np.random.choice(real_idx, n_trials, replace=False)
        fake_idx = np.random.choice(fake_idx, n_trials, replace=False)
        
        for i in range(n_trials):
            # print(real_X.shape, real_X[real_idx[i], :, :, None].shape)
            single_stft(real_X[[real_idx[i]]], fake_X[[fake_idx[i]]], f'stft_{key}_{i}', 
                       channels, fs, path=path)
            
            spectrum, _ = plot_spectrum(real_X[[real_idx[i]]], fake_X[[fake_idx[i]]],
                                        fs, f'{key}_{i}')

            spectrum.savefig(os.path.join(path,  f'spectrum_{key}_{i}'), dpi=300)
            plt.close('all')




if __name__ == "__main__":
    save_path = f'/home/boehms/eeg-gan/EEG-GAN/temp_plots/single_spectrum'

    plot_single_trials(batch.real, batch.y_real, batch.fake, 
                       batch.y_fake, mapping, fs, channels, 15, save_path)
    
    spectrum, _ = plot_spectrum(batch.real, batch.fake, fs, f'full batch')
    spectrum.savefig(os.path.join(save_path,  f'FULL_SPECTRUM'), dpi=300)

    # Plot Full spectrum: 
    real_stft, fake_stft, _, f, t = calc_bin_stats(batch.real,  batch.fake, fs)

    plot_bin_stats(batch.real, batch.fake, fs, channels, save_path, 'FULL_BIN_STATS')

    for key in mapping.keys():
            conditional_real = batch.real[batch.y_real == mapping[key]]
            conditional_fake = batch.fake[batch.y_fake == mapping[key]]

            # calculate frequency in current stage:   
            plot_bin_stats(conditional_real,conditional_fake,
                            fs, channels, save_path, str(key))


