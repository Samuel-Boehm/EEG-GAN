# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kstest
from statsmodels.stats.multitest import fdrcorrection


def compute_stft(batch: np.ndarray, fs: float, wsize: int):
    """
    compute the short time fourier tramsform for a 
    batch of trials (trials x channels x timepoints). 


    Arguments:
    ----------
        batch (ndarray): batch of data with shape 
        trials x channels x timepoints
        
        fs (float): sampling frequency of data
        
        wsize (int): window size for stft

    Returns:
    ----------
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

def plot_stft(stft_power: np.ndarray, t: np.ndarray, f: np.ndarray, title: str, 
            cbar_label: str, channel_names: list=None, upper = 5, lower = -5, 
            cmap = 'RdBu_r', path:str = None):
    
    """
    Plot multi channel time-frequency-power spectrum from stft for 21 channels.

    Arguments:
    ----------
        stft_power (ndarray): Array of stft-power calculations of 
        shape channels x freq_window x time_window
        
        t (ndarray): Array of segment times.
        
        f (ndarray): Array of sample frequencies.
        
        title (str): Plot title
        
        cbar_label (str): Label for colorbar
        
        channel_names (list(str), optional): list of Channel names. Defaults to empty None
    Returns:
    ----------
        figure (figure): figure object
    """
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(3, 7,  figsize=(13*cm, 13*cm), facecolor='w')
        
        
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
            y_ticks_ = np.array([2, 6, 10, 21, int((21+pos.max()) / 2 )])
            y_ticks = y_ticks_[y_ticks_ <= pos.max()]

            y_label = np.array([r'$ \delta $', r'$ \theta $' ,r'$ \alpha $', r'$ \beta $', r'$ \gamma $'])
            y_label = y_label[y_ticks_ <= pos.max()]
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

    # figure = plt.gcf()  # get current figure
    fig.set_size_inches(40*cm, 20*cm)
    fig.subplots_adjust(right=0.87, top=.95)
    cbar_ax = fig.add_axes([0.35, 0.01, 0.3, 0.01])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=0, fontsize=12, labelpad=5)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.rcParams['savefig.facecolor']='white'
    if path:
        plt.savefig(path, format='png', bbox_inches='tight')

    return fig, axs
    


def blc(stft_array: np.ndarray, t: np.ndarray, blc_win: tuple = None):

    '''
    Applies baseline corretion of a batch of 
    Short Time Fourier Transformations(STFT).

    Arguments:
    ----------
        stft_array : array of short time fourier transormations of shape
            trials x channels x freq_window x time_window

        t : array of segment times (as returned from scipy.signal.stft)

        blc_win : start and stop time as reference for baseline corretion
            if None: whole timeframe is taken for blc

    Returns:
    ----------
        med_log : median of the relative power over all trials with base
        10 logarithm applied. Shape: channels x freq_window x time_window
        
    '''

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


def calc_bin_stats(real_X: np.ndarray, fake_X: np.ndarray, fs: int,):
    '''
    Calculates the p-values for each bin of the stft and corrects them for multiple comparisons.
    Returns the corrected p-values reshaped to the shape of the stft. Bins are considered to be 
    similar if the p-value is bigger than 0.3.

    Arguments:
    ----------
        real_X (ndarray): Array of real data with shape trials x channels x timepoints
        
        fake_X (ndarray): Array of fake data with shape trials x channels x timepoints
        
        fs (int): Sampling frequency of the data

    Returns:
    ----------
        median_real (ndarray): Array of stft for real data
        
        shape: channels x freq_window x time_window
        
        median_fake (ndarray): Array of stft for fake data
        
        shape: channels x freq_window x time_window
        
        corr_pval (ndarray): Array of corrected p-values with shape freq_window x time_window

        f (ndarray): Array of sample frequencies.

        t (ndarray): Array of segment times.
    
    '''

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
        
    real_stft = np.mean(real_stft, axis=0)
    fake_stft = np.mean(fake_stft, axis=0)
  
    return real_stft, fake_stft, corr_pval, f, t


def calc_bin_stats_for_mapping(real_X, real_y, fake_X, fake_y, fs: int,
                         mapping: dict):
    '''
    Calculates the p-values for each bin of the stft and corrects them for multiple comparisons.
    If an out path is provided all plots are saved there

    Arguments:
    ----------
        real_X (ndarray): Array of real data with shape trials x channels x timepoints

        real_y (ndarray): Array of real labels with shape trials x 1

        fake_X (ndarray): Array of fake data with shape trials x channels x timepoints

        fake_y (ndarray): Array of fake labels with shape trials x 1

        fs (int): Sampling frequency of the data

        mapping (dict): Dictionary with mapping of labels to conditions
    
    Returns:
    ----------
        resuts_dict (dict): Dictionary with ffts for each condition on real and fake
        dict[condition] = [real_stft, fake_stft, p_vals, f, t]
    '''

    # Creating a dictionary with real and fake data for each condition 
    # dict[condition] = [real_data, fake_data]

    conditional_dict = {}

    for key in mapping.keys():
        conditional_dict[key] = [
            real_X[real_y == mapping[key]],
            fake_X[fake_y == mapping[key]]
            ]
    
    # Dictionary with ffts for each condition on real and fake
    resuts_dict = {}

    for key in mapping.keys():
        real_stft, fake_stft, p_vals, f, t, = calc_bin_stats(conditional_dict[key][0], conditional_dict[key][1], fs)
        resuts_dict[key] = [real_stft, fake_stft, p_vals, f, t]
    
    return resuts_dict
   
def plot_bin_stats(real_X: np.ndarray, fake_X, fs, channels:list, path:str=None,
                   title:str='plot', show_fig:bool=False):
    '''
    Plots the bin statistics of the real and fake data.

    Arguments:
    ----------
    real_X: np.ndarray
        The real data.
    fake_X: np.ndarray
        The fake data.
    fs: float
        Sampling frequency of the data.
    channels: list
        The names of the channels.
    path: str
        The path where the plot should be saved.
    title: str
        The title of the plot.
    show_fig: bool
        If True the plot is shown.

    Returns:
    ----------
        Tuple of figures (fig_stats, fig_real, fig_fake)

    '''
    
    real_stft, fake_stft, pvals, f, t = calc_bin_stats(real_X, fake_X, fs)

    color_grid = np.zeros_like(pvals)

    # If p-value bigger 0.3 we plot the log10 rel. power of the fake data median
    color_grid[pvals >= 0.3] =  fake_stft[pvals >= 0.3]
    
    # Else we plot a black x on grey background
    color_grid[pvals < 0.3] =  np.nan
    color_grid = np.ma.masked_invalid(color_grid)

    if path:
        path_stats=os.path.join(path, f'{title}_stats_{fs}.png')
        path_real=os.path.join(path, f'{title}_real_{fs}.png')
        path_fake=os.path.join(path, f'{title}_fake_{fs}.png')
    else:
        path_stats = None
        path_real = None
        path_fake = None
                          



    fig_stats, _ = plot_stft(
            color_grid, t, f, f'{title} {fs} Hz - p value > 0.3', 'log10 rel. power mean', channel_names=channels,
            upper=5, lower = -5, path=path_stats,
            )
    

    fig_real, _ = plot_stft(
            real_stft, t, f, f'{title} real {fs} Hz ', 'log10 rel. power mean',
            channel_names=channels, upper=5, lower = -5, path=path_real,
            )

    fig_fake, _ = plot_stft(
            fake_stft, t, f, f'{title} fake {fs} Hz ', 'log10 rel. power mean',
            channel_names=channels, upper=5, lower = -5, path=path_fake,
            )
    
    return fig_stats, fig_real, fig_fake
