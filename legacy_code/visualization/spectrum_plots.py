# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
from legacy_code.visualization.utils import labeled_tube_plot

def plot_spectrum(real: np.ndarray, fake: np.ndarray = None, fs = 256, name = ''):
    '''
    Plot the spectrum of the real and fake data
    Args:
        real: real data
        fake: fake data
    '''

    freqs = np.fft.rfftfreq(real.shape[2], 1. / fs)

    amplitudes_real = np.log(np.abs(np.fft.rfft(real, axis=2)))
    real_mean = amplitudes_real.mean(axis=(0, 1)).squeeze()
    real_std = amplitudes_real.std(axis=(0, 1)).squeeze()
        
    aplitudes_fake = np.log(np.abs(np.fft.rfft(fake, axis=2)))
    fake_mean = aplitudes_fake.mean(axis=(0, 1)).squeeze()
    fake_std = aplitudes_fake.std(axis=(0, 1)).squeeze()
        


    figure, ax = labeled_tube_plot(freqs,
                      [real_mean, fake_mean],
                      [real_std, fake_std],
                      ["Real", "Fake"],
                      f"Mean spectral log amplitude {name}", "Hz", "log(Amp)", None)
    
    
    ax.text(.01, .99, f'n: {str(real.shape[0])} - fs: {fs} - data shape: {fake.shape}', fontsize=12, ha='left', va='top', transform=ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 4})
        
    
    return figure, ax