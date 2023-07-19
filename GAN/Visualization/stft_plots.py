import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kstest
from statsmodels.stats.multitest import fdrcorrection

def compute_stft(data, fs, wsize):
    '''
    Compute the STFT of the data
    Args:
        data: data to compute the STFT. shape: (batch_size, channels, time)
        fs: sampling frequency of the data
        wsize: window size in samples, 0.2 - 0.5 * fs is a good value
    Returns:
        stft_arr: magnitude of the data. shape: (batch_size, channels, freqs, time)
        f: frequencies. used for plotting. unit: Hz
        t: time. used for plotting. unit: s
    '''

    # choose the overlap size as needed
    noverlap = wsize * 3 / 4
    
    stft_arr = []
    
    for sample in data:
        f, t, Zxx = signal.stft(sample, fs, nperseg=wsize, noverlap=noverlap,
            padded=False, boundary=None, window="hann", )
                
        stft_arr.append(Zxx)

    stft_arr = np.asarray(np.abs(stft_arr) ** 2)

    return stft_arr, f, t

def blc(stft_array: np.ndarray, t: np.ndarray, blc_win: tuple = None):

    '''
    Applies baseline corretion of a batch of 
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
    med_log = 10 * np.log10(stft_blc)
    
    return med_log



def plot_stft(stft_power: np.ndarray, t: np.ndarray, f: np.ndarray,
              title : str, cbar_label : str, channel_names: list):
    '''
    Plot the STFT per channel.
    Args:
        stft_power: STFT of the data. shape: (channels, freqs, time)
        t: time array.
        f: frequencie array.
        new_title: title of the plot.
        cbar_label: label of the colorbar.
        channel_names: names of the channels.
    '''
    _, axs = plt.subplots(3, 7,  figsize=(30, 10), facecolor='w')
    
    for i, ax in enumerate(axs.flatten()):
        pcm = ax.pcolormesh(
            t - .5,
            f,
            stft_power[i],
            cmap="RdBu_r",
            vmax=5., vmin=-5.,
        )

        ax.text(3, 60, channel_names[i],
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    
    plt.grid(False)

    if title != '':
        plt.suptitle(title, fontsize=30,)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18)
    # set figure's size manually to your full screen (32x18)
    figure.subplots_adjust(right=0.87, top=.95)
    cbar_ax = figure.add_axes([0.89, 0.4, 0.004, 0.3])
    cbar = figure.colorbar(pcm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, rotation=-270, fontsize=16, labelpad=10)
    plt.rcParams['savefig.facecolor']='white'
    plt.savefig('stft.png', dpi=300, bbox_inches='tight')

def compute_median(stft_arr:np.ndarray, t:np.ndarray):
    '''
    Compute the median and apply baseline correction.
    Args:
        stft_arr: STFT of the data. shape: (batch_size, channels, freqs, time)
        t: time array.
    '''
    # define interval for baseline correction (blc):
    # Only take first 0.5 seconds for blc
    blc_win = t <= 0.5

    avg = np.median(stft_arr, axis=0) # along batches
    # take the mean of the median
    avg_mean = np.mean(avg[:,:,blc_win], axis=-1) # along time
    # rescale by mean
    avg_reshaped = np.expand_dims(avg_mean, axis=-1)
    
    avg_reshaped = np.broadcast_to(
        avg_reshaped,
        shape = (
            avg_mean.shape[0],
            avg_mean.shape[1],
            stft_arr.shape[-1],
        ),
    )

    # expand to trials
    avg_reshaped = np.broadcast_to(
        avg_reshaped,
        (stft_arr.shape[0],
        avg_reshaped.shape[0],
        avg_reshaped.shape[1],
        avg_reshaped.shape[2],),)

    stft_arr = np.divide(stft_arr, avg_reshaped)
    del avg_reshaped, avg_mean
    # # take the median of bl corrected epochs
    stft_med = np.median(stft_arr, axis=0)
    med_log = 10 * np.log10(stft_med)
    del stft_arr, stft_med
    return med_log

def calculate_statistics(real, fake, fs):
    '''
    Calculate the bin statistics between real and fake stft data.
    Args:
        real: real data. shape: (trials, channels, time)
        fake: fake data. shape: (trials, channels, time)
        fs: sampling frequency.
        
    Returns:
        median_real: baseline corrected stft median of the real data. shape: (channels, freqs, time)
        median_fake: baseline corrected stft median of the fake data. shape: (channels, freqs, time)
        colr_grid: color grid for the plot. Plot tis to visualize statistics
        shape: (channels, freqs, time)
        t: time array.
        f: frequency array.
    '''

    # Setting frequency and window size

    wsize = fs/2
    
    
    # Dictionary with ffts for each condition on real and fake
    
    fft_real, f, t = compute_stft(real, fs, wsize)
    fft_fake, _, _ = compute_stft(fake, fs, wsize)    

    # Baseline correction with pre-stimulus data as bl (first 0.5 seconds of trial)
    
    blc_real = blc(fft_real, t, (0, .5))
    blc_fake = blc(fft_fake, t, (0, .5))
    
    
    
    # flatten out data to have shape: n_samples x (time_bins x freq_bins)
    blc_real_ = blc_real.reshape(blc_real.shape[0], np.prod(blc_real.shape[1:]))
    blc_fake_ = blc_fake.reshape(blc_fake.shape[0], np.prod(blc_fake.shape[1:]))

    p_values = []

    for i in range(blc_real_.shape[1]):
        p_values.append(kstest(blc_real_ [:, i], blc_fake_ [:, i])[1])
        
    _, corr_pval = fdrcorrection(p_values, alpha=0.3, method='n')

    pvals = corr_pval.reshape(blc_real.shape[1:])
    
    # create similarity grid:
    color_grid = np.zeros_like(pvals)

    # If p-value bigger 0.3 we plot the log10 rel. power of the fake data median
    median = np.median(blc_fake, axis=0)
    color_grid[pvals >= 0.3] =  median[pvals >= 0.3]
    
    # Else we plot a black x on grey background
    color_grid[pvals < 0.3] =  np.nan
    color_grid = np.ma.masked_invalid(color_grid)

    median_real = np.median(blc_real, axis=0)
    median_fake = np.median(blc_fake, axis=0)

    return median_real, median_fake, color_grid, t, f
