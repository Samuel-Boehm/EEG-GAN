import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
        f, t, Zxx = signal.stft(sample, fs,
            nperseg=wsize, noverlap=noverlap,
            padded=False, boundary=None,
            window="hann", )
        
        stft_arr.append(Zxx)

    stft_arr = np.asarray(np.abs(stft_arr) ** 2)

    return stft_arr, f, t


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
            vmax=5., vmin=-5.
        )

        ax.text(3, 60, channel_names[i],
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    
    plt.grid(False)

    if title != '':
        plt.suptitle(
            title,
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
    
def compute_median(stft_arr:np.ndarray, t:np.ndarray):
    '''
    Compute the median and apply baseline correction.
    Args:
        stft_arr: STFT of the data. shape: (batch_size, channels, freqs, time)
        t: time array.
    '''
    # define intervall for baseline correction (blc):
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
        (
            stft_arr.shape[0],
            avg_reshaped.shape[0],
            avg_reshaped.shape[1],
            avg_reshaped.shape[2],
        ),
    )

    stft_arr = np.divide(stft_arr, avg_reshaped)
    del avg_reshaped, avg_mean
    # # take the median of bl corrected epochs
    stft_med = np.median(stft_arr, axis=0)
    med_log = 10 * np.log10(stft_med)
    del stft_arr, stft_med
    return med_log
