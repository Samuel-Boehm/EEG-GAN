# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def computeStft(epochs_left, fs, wsize):

# choose the overlap size as needed
    noverlap = wsize * 3 / 4

    temp_stft_arr = []
    for row in epochs_left:

        f, t, Zxx = signal.stft(
            row,
            fs,
            nperseg=wsize,
            noverlap=noverlap,
            padded=False,
            boundary=None,
            window="hann",
        )
        temp_stft_arr.append(Zxx)

    stft_arr = np.asarray(np.abs(temp_stft_arr) ** 2)

    del temp_stft_arr

    return stft_arr, f, t

def plotStft(stft_power, t, f, new_title, cbar_label, channel_names):
    _, axs = plt.subplots(3, 7,  figsize=(30, 10), facecolor='w')
    
    for i, ax in enumerate(axs.flatten()):
        pcm = ax.pcolormesh(
            t - .5,
            f,
            stft_power[i],
            cmap="RdBu_r",
            vmax=5., vmin=-5.
        )

        ax.text(3, 60, channel_names[i],
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    
    plt.grid(False)

    if new_title != '':
        plt.suptitle(
            new_title,
            fontsize=30,
        )

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18)
    # set figure's size manually to your full screen (32x18)
    figure.subplots_adjust(right=0.87, top=.95)
    cbar_ax = figure.add_axes([0.89, 0.4, 0.004, 0.3])
    cbar = figure.colorbar(pcm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=-270, fontsize=16, labelpad=10)
    plt.rcParams['savefig.facecolor']='white'
    plt.show()

def computeMedian(stft_arr2, t):
    # define intervall for baseline correction (blc):
    # Only take first 0.5 seconds for blc
    blc_win = t <= 0.5

    avg = np.median(stft_arr2, axis=0)
    # take the mean of the median
    # stft_arr2 = data
    avg_mean = np.mean(avg[:,:,blc_win], axis=-1)
    # rescale by mean
    avg_reshaped = np.expand_dims(avg_mean, axis=-1)
    avg_reshaped = np.broadcast_to(
        avg_reshaped,
        (
            avg_mean.shape[0],
            avg_mean.shape[1],
            # avg_mean.shape[2],
            stft_arr2.shape[-1],
        ),
    )

    # expand to trials
    avg_reshaped = np.broadcast_to(
        avg_reshaped,
        (
            stft_arr2.shape[0],
            avg_reshaped.shape[0],
            avg_reshaped.shape[1],
            avg_reshaped.shape[2],
            # avg_reshaped.shape[3],
        ),
    )

    stft_arr3 = np.divide(stft_arr2, avg_reshaped)
    del avg_reshaped, stft_arr2, avg_mean
    # # take the median of bl corrected epochs
    stft_med = np.median(stft_arr3, axis=0)
    # stft_med = np.mean(stft_arr3, axis=0)
    med_log = 10 * np.log10(stft_med)
    del stft_arr3, stft_med
    return med_log
