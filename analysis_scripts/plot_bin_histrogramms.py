# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.visualization.stft_plots import calc_bin_stats_for_mapping, blc, compute_stft
from scipy.stats import kstest
from statsmodels.stats.multitest import fdrcorrection
from gan.visualization.spectrum_plots import plot_spectrum
from gan.data.batch import batch_data
from gan.paths import data_path
import numpy as np
from gan.utils import generate_data
from gan.data.utils import downsample_step
import matplotlib.pyplot as plt

import torch
import os

import sys


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

# Subsample the bigger dataset:
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


def stft(real_X: np.ndarray, fake_X: np.ndarray, fs: int,):
    # Setting window size
    wsize = fs/2
    # calculate short time fourier transform
    real_stft, _ ,_ = compute_stft(real_X, fs, wsize)
    fake_stft, f, t = compute_stft(fake_X, fs, wsize)
    # Baseline correction with pre-stimulus data as bl (here first 0.5 seconds of trial)

    real_stft = blc(real_stft, t, (0, .5))
    fake_stft = blc(fake_stft, t, (0, .5))

    # flatten out data to have shape: n_samples x (time_bins * freq_bins)
    real_stft_ = real_stft.reshape(real_stft.shape[0], np.prod(real_stft.shape[1:]))
    fake_stft_ = fake_stft.reshape(fake_stft.shape[0], np.prod(fake_stft.shape[1:]))

    p_values = []

    for i in range(real_stft_.shape[1]):
        p_values.append(kstest(real_stft_ [:, i], fake_stft_ [:, i])[1])
        
    _, corr_pval = fdrcorrection(p_values, alpha=0.3, method='n')

    corr_pval = corr_pval.reshape(real_stft.shape[1:])
  
    return real_stft, fake_stft, corr_pval, f, t


def apply_stft_on_mapping(real_X, real_y, fake_X, fake_y, fs: int,
                         mapping: dict):
   
    conditional_dict = {}

    for key in mapping.keys():
        conditional_dict[key] = [
            real_X[real_y == mapping[key]],
            fake_X[fake_y == mapping[key]]
            ]
    
    # Dictionary with ffts for each condition on real and fake
    resuts_dict = {}

    for key in mapping.keys():
        real_stft, fake_stft, p_vals, f, t, = stft(conditional_dict[key][0], conditional_dict[key][1], fs)
        resuts_dict[key] = [real_stft, fake_stft, p_vals, f, t]
    
    return resuts_dict


if __name__ == "__main__":
    save_path = f'/home/boehms/eeg-gan/EEG-GAN/temp_plots/bin_spectrum/'


    stats_dict = apply_stft_on_mapping(batch.real, batch.y_real, batch.fake, batch.y_fake,
                                            fs, mapping)

    for key in mapping.keys():
        real_stft, fake_stft, p_vals, _, _  = stats_dict[key]
        print('real stft shape: ', real_stft.shape)
        
        for ch in range(real_stft.shape[-1]):
            fig, axs = plt.subplots(real_stft.shape[1], real_stft.shape[2], figsize=(200, 100))
            for i in range(real_stft.shape[1]):
                for j in range(real_stft.shape[2]):
                    axs[i, j].hist(real_stft[:, i, j, 0], bins=100, alpha=0.5, label='real')
                    axs[i, j].hist(fake_stft[:, i, j, 0], bins=100, alpha=0.5, label='fake')
                    axs[i, j].vlines(np.mean(real_stft[:, i, j, 0]), 0, 50, color='blue', label='real mean')
                    axs[i, j].vlines(np.mean(fake_stft[:, i, j, 0]), 0, 50, color='orange', label='fake mean')

                    axs[i, j].text(.94, .96, str(round(p_vals[i-1, j-1, 0], 2)),
                                    fontsize=12, ha='right', va='top', transform=axs[i, j].transAxes, 
                                    bbox={
                                        'facecolor': ['green' if p_vals[i-1, j-1, 0] > 0.3 else 'red'][0],
                                        'alpha': 0.8,
                                        'pad': 4})

            fig.savefig(os.path.join(save_path, f'bin_histogramm_{key}_{str(ch)}.png'))
            plt.close(fig)
        sys.exit()
