#  Author: Samuel Boehm <samuel-boehm@web.de>
import sys
# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.data.dataset import Data
from eeggan.data.preprocess.resample import downsample
from utils import create_balanced_datasets


from scipy import signal
from scipy.stats import kstest
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pickle
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--gan_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Results/Thesis/')
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--stage', type=int, required=True)
parser.add_argument('--subject', type=int, required=False)
parser.add_argument('--data_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Data/Thesis/')
parser.add_argument('--deep4_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Models/Thesis/')
parser.add_argument('--downsample', type=bool, required=False, default=False)

args = parser.parse_args()

gan_path = os.path.join(args.gan_path, args.name)

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'M1', 'M2']
MAPPING = {'right': 0, 'rest': 1}


def compute_stft(batch: np.ndarray, fs: float, wsize: int):
    """
    compute the short time fourier tramsform for a 
    batch of trials (trials x channels x timepoints). 


    Args:
        batch (ndarray): batch of data with shape 
        trials x channels x timepoints
        
        fs (float): sampling frequency of data
        
        wsize (int): window size for stft

    Returns:
        stft_arr (ndarray): array of stft for each channel in each trial
        with shape trials x channels x freq_window x time_window
        
        f (ndarray): Array of sample frequencies.

        t (ndarray): Array of segment times.
    """

    # choose the overlap size as needed
    noverlap = wsize * 3 / 4

    temp_stft_arr = []
    for trial in batch:
        f, t, Zxx = signal.stft(
            trial,
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

def plot_stft(
            stft_power: np.ndarray, t: np.ndarray, f: np.ndarray, title: str, 
            cbar_label: str, channel_names: list=None, upper = 5, lower = -5, 
            cmap = 'RdBu_r', path:str = None):
    """Plot multi channel time-frequency-power spectrum from stft for 21 channels.

        Args:
            stft_power (ndarray): Array of stft-power calculations of 
            shape channels x freq_window x time_window
            
            t (ndarray): Array of segment times.
            
            f (ndarray): Array of sample frequencies.
            
            title (str): Plot title
            
            cbar_label (str): Label for colorbar
            
            channel_names (list(str), optional): list of Channel names. Defaults to empty None
        """
    cm = 1/2.54  # centimeters in inches
    _, axs = plt.subplots(3, 7,  figsize=(13*cm, 13*cm), facecolor='w')
        
        
    for i, ax in enumerate(axs.flatten()):
        pcm = ax.pcolormesh(
            t - .5,
            f,
            stft_power[i],
            cmap=cmap,
            vmax=upper, vmin=lower
        )
        ax.patch.set(hatch='xx', edgecolor='darkgrey', facecolor='lightgrey')
        
        if channel_names:
            ax.text(.94, .96, channel_names[i], fontsize=12, ha='right', va='top', transform=ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 4})
        
        # Drawing the frequency bands and label it on y axis instead of frequencies
        pos = ax.get_yticks()
        label = ax.get_yticks()
        
        frequency_bands = np.array([0, 4, 8, 12, 30])
        frequency_bands = frequency_bands[frequency_bands <= pos.max()]
        for line in frequency_bands:
            ax.axhline(y=line, color='k', lw=1, ls=':')

        if i % 7 == 6:
            y_ticks = np.array([2, 6, 10, 21, int((21+pos.max()) / 2 )])
            y_ticks = y_ticks[y_ticks <= pos.max()]

            y_label = np.array([r'$ \delta $', r'$ \theta $' ,r'$ \alpha $', r'$ \beta $', r'$ \gamma $'])
            y_label = y_label[y_ticks <= pos.max()]
            ax.yaxis.tick_right()
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_label, fontsize=10)
            ax.yaxis.set_label_position("right")
            if i == 13:
                ax.set_ylabel('Frequency band', fontsize=12)
        
        elif i % 7 > 0:
            ax.set_yticks([])
        
        else:
            if i ==7:
                ax.set_ylabel('Hz', fontsize=12)

        if i // 7 != 2:
            ax.set_xticks([])
        else:
            if i == 17:
                ax.set_xlabel('time [s]', fontsize=12)

        
    plt.grid(False)

    if title != '':
        plt.suptitle(
            title,
            fontsize=14,
        )

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(40*cm, 20*cm)
    figure.subplots_adjust(right=0.87, top=.95)
    cbar_ax = figure.add_axes([0.35, 0.01, 0.3, 0.01])
    cbar = figure.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=0, fontsize=12, labelpad=5)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.rcParams['savefig.facecolor']='white'
    if path:
        plt.savefig(path, format='pdf', bbox_inches='tight')
    



# Test Kolmogorow-Smirnow unsigned if p > 0.3 its eveidence for same distribution
def blc(stft_array: np.ndarray, t: np.ndarray, blc_win: tuple = None):

    r"""Applies baseline corretion of a batch of 
    Short Time Fourier Transformations(STFT).

    Parameters
    ----------
    stft_array : array of short time fourier transormations of shape
    trials x channels x freq_window x time_window

    t : array of segment times (as returned from scipy.signal.stft)

    blc_win : start and stop time as reference for baseline corretion
    if None: whole timeframe is taken for blc

    Returns
    -------
    med_log : median of the relative power over all trials 
    with base 10 logarithm applied. Shape: channels x freq_window x time_window
        
    """

    # define intervall for baseline correction (blc):
    
    
    if blc_win:
        assert blc_win[0] <= blc_win[1],(
        f' {blc_win[0]} >= {blc_win[1]}. First time point needs to be be smaller than second'
        )
        
        win = np.logical_and(t >= blc_win[0], t <= blc_win[1])
    else:
        win = np.ones(t.shape, dtype=bool)

    # 1 take median over trials -> channels x freq_window x time_window
    median = np.median(stft_array, axis=0)

    # 2 take mean over blc cutout ->  channels x freq_window
    baseline = np.mean(median[:,:,win], axis=-1)

    # rescale: fill each time bin with the baseline and expand
    # to trials -> trials x channels x freq_window x time_window
    baseline_reshaped = np.expand_dims(baseline, axis=-1)
    baseline_reshaped = np.broadcast_to(baseline_reshaped,
        (
            stft_array.shape[0],
            baseline.shape[0],
            baseline.shape[1],
            stft_array.shape[-1],
        ),
    )

    # 3 Correct by basline:
    stft_blc = np.divide(stft_array, baseline_reshaped)

    # gc
    del baseline_reshaped, stft_array, baseline

    # no median taken (step 4)

    # 5 take the base 10 logarithm
    med_log = 10 * np.log10(stft_blc + 10e-8)
    
    return med_log



def plot_time_domain(real, fake, channel_names, path):
    
    cm = 1/2.54  # centimeters in inches
    _, axs = plt.subplots(3, 7,  figsize=(40*cm, 20*cm), facecolor='w')
    x = np.linspace(-.5, 4, num = fake.shape[2])

    for i, ax in enumerate(axs.flatten()):
        if channel_names:
            ax.text(.94, .94, channel_names[i], fontsize=12, ha='right', va='top', transform=ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 4})
        
        real_mean = real.mean(axis = 0)
        fake_mean = fake.mean(axis = 0)

        stack = np.concatenate((real_mean, fake_mean))
        ax.plot(x, real_mean[i], c='#c1002a', label ='real', lw = 1)
        ax.plot(x, fake_mean[i], c='#004a99', label='fake', lw = 1)
        
        max_y = stack.max() + stack.max() * .1
        min_y = stack.min() + stack.min() * .1

        ax.set_ylim(bottom=min_y, top=max_y)
        # ax.fill_between(x, fake.mean(axis = 0)[i] + fake.std(axis = 0)[i], fake.mean(axis = 0)[i] - fake.std(axis = 0)[i], color='#004a99', alpha=.4)
        # ax.fill_between(x, real.mean(axis = 0)[i] + real.std(axis = 0)[i], real.mean(axis = 0)[i] - real.std(axis = 0)[i], color='#c1002a', alpha=.4)

        if i % 7 > 0:
            ax.set_yticks([])
        
        else:
            if i ==7:
                ax.set_ylabel('Amplitude', fontsize=12)

        if i // 7 != 2:
            ax.set_xticks([])
        else:
            if i == 17:
                ax.set_xlabel('time [s]', fontsize=12)

    plt.legend(loc=4)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if path:
        plt.savefig(path, format='pdf', bbox_inches='tight')

def calculate_statistics(real: Data, fake: Data, stage: int,
                         mapping: dict, out_path: str=None, show_plots=True):
    '''If an out path is provided all plots are saved there'''

    # Creating a dictionary with real and fake data for each condition 
    # dict[condition] = [real_data, fake_data]

    conditional_dict = {}

    for key in mapping.keys():
        conditional_dict[key] = [
            real.X[real.y == mapping[key]],
            fake.X[fake.y == mapping[key]]
            ]


    # Setting frequency and window size
    fs = 8 * 2**stage
    wsize = fs/2
    
    
    # Dictionary with ffts for each condition on real and fake
    fft_dict = {}

    for key in mapping.keys():
        fft_dict[key]= [
            compute_stft(conditional_dict[key][0], fs, wsize)[0],
            compute_stft(conditional_dict[key][1], fs, wsize)[0],
        ]

    # Calculate stft once for f and  t
    _, f, t = compute_stft(conditional_dict[key][1], fs, wsize)

    # Baseline correction with pre-stimulus data as bl (first 0.5 seconds of trial)
    blc_dict = {}
    
    for key in mapping.keys():
        blc_dict[key] = [
            blc(fft_dict[key][0], t, (0, .5)),
            blc(fft_dict[key][1], t, (0, .5))
        ]
    
    pval_dict = {}
    for key in mapping.keys():
        blc_real = blc_dict[key][0]
        blc_fake = blc_dict[key][1]

        # flatten out data to have shape: n_samples x (time_bins x freq_bins)
        blc_real_ = blc_real.reshape(blc_real.shape[0], np.prod(blc_real.shape[1:]))
        blc_fake_ = blc_fake.reshape(blc_fake.shape[0], np.prod(blc_fake.shape[1:]))

        p_values = []

        for i in range(blc_real_.shape[1]):
            p_values.append(kstest(blc_real_ [:, i], blc_fake_ [:, i])[1])
            
        _, corr_pval = fdrcorrection(p_values, alpha=0.3, method='n')

        corr_pval = corr_pval.reshape(blc_real.shape[1:])
        
        pval_dict[key] = corr_pval
    
    # Plot similarity for each condition:
    for key in mapping.keys():
        pvals = pval_dict[key]
        color_grid = np.zeros_like(pvals)

        # If p-value bigger 0.3 we plot the log10 rel. power of the fake data median
        median = np.median(blc_dict[key][1], axis=0)
        color_grid[pvals >= 0.3] =  median[pvals >= 0.3]
        
        # Else we plot a black x on grey background
        color_grid[pvals < 0.3] =  np.nan
        color_grid = np.ma.masked_invalid(color_grid)

        plot_stft(
                color_grid, t, f, f'{key} {fs} Hz - p value > 0.3', 'log10 rel. power median', channel_names=CHANNELS,
                upper=5, lower = -5, path=os.path.join(out_path, f'{key}_stats_{stage}.pdf')
                )
        
        median_real = np.median(blc_dict[key][0], axis=0)
        median_fake = np.median(blc_dict[key][1], axis=0)
        

        plot_stft(
                median_real, t, f, f'{key} real {fs} Hz ', 'log10 rel. power median',
                channel_names=CHANNELS, upper=5, lower = -5, path=os.path.join(out_path, f'{key}_real_{stage}.pdf')
                )

        plot_stft(
                median_fake, t, f, f'{key} fake {fs} Hz ', 'log10 rel. power median',
                channel_names=CHANNELS, upper=5, lower = -5, path=os.path.join(out_path, f'{key}_fake_{stage}.pdf')
                )

        plot_time_domain(conditional_dict[key][0], conditional_dict[key][1], channel_names=CHANNELS, path=os.path.join(out_path, f'{key}_TD_{stage}'))

    return pval_dict


if __name__ == '__main__':
    if args.name == 'torch_cGAN': 
        real = joblib.load(os.path.join(args.data_path, f'tGAN_real_stage{args.stage}.dataset'))
        fake = joblib.load(os.path.join(args.data_path, f'tGAN_fake_stage{args.stage}.dataset'))

    else:
        real, fake = create_balanced_datasets(gan_path, args.data_path, args.data_name, args.stage, None, False, 3000)
    
    if args.downsample == True:
        real.X = downsample(real.X, 2, axis=2)
        fake.X = downsample(fake.X, 2, axis=2)

    out_path = f'/home/boehms/eeg-gan/EEG-GAN/EEG-GAN/analysis_scripts/outputs/{args.name}'

    os.makedirs(out_path, exist_ok=True)
    
    p_vals =  calculate_statistics(
                real, 
                fake,
                args.stage,
                MAPPING, 
                out_path=out_path)

    with open(os.path.join(out_path, f'{args.name}_stage{args.stage}_pvals.pkl'), 'wb') as f:
        pickle.dump(p_vals, f)
