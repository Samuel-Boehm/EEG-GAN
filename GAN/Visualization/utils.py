import matplotlib.pyplot as plt
import numpy as np


def labeled_tube_plot(x, data_y, tube_y, labels,
                      title="", xlabel="", ylabel="", axes=None,  autoscale = True):
    x = np.asarray(x)
    data_y = np.asarray(data_y)
    tube_y = np.asarray(tube_y)

    if axes is None:
        figure, axes = plt.subplots()

    colors = []
    for i, label in enumerate(labels):
        y_tmp = data_y[i]
        tube_tmp = tube_y[i]
        p = axes.fill_between(x, y_tmp + tube_tmp, y_tmp - tube_tmp, alpha=0.5, label=labels[i])
        colors.append(p._original_facecolor)

    for i, label in enumerate(labels):
        axes.plot(x, data_y[i], lw=2, color=colors[i])

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    if autoscale:
        axes.set_ylabel(ylabel)
        axes.set_xlim(x.min(), x.max())
        axes.set_ylim(np.nanmin(data_y - tube_y), np.nanmax(data_y + tube_y))
    axes.legend()

    return figure, axes



def plot_spectrum(batch_real: np.ndarray, batch_fake: np.ndarray = None, fs = 256, name = ''):
    '''
    Plot the spectrum of the real and fake data
    Args:
        batch_real: real data
        batch_fake: fake data
    '''

    freqs = np.fft.rfftfreq(batch_real.shape[2], 1. / fs)

    amplitudes_real = np.log(np.abs(np.fft.rfft(batch_real, axis=2)))
    real_mean = amplitudes_real.mean(axis=(0, 1)).squeeze()
    real_std = amplitudes_real.std(axis=(0, 1)).squeeze()
        
    aplitudes_fake = np.log(np.abs(np.fft.rfft(batch_fake, axis=2)))
    fake_mean = aplitudes_fake.mean(axis=(0, 1)).squeeze()
    fake_std = aplitudes_fake.std(axis=(0, 1)).squeeze()
        


    figure, ax = labeled_tube_plot(freqs,
                      [real_mean, fake_mean],
                      [real_std, fake_std],
                      ["Real", "Fake"],
                      f"Mean spectral log amplitude {name}", "Hz", "log(Amp)", None)
    
    
    ax.text(.01, .99, f'n: {str(batch_real.shape[0])} - fs: {fs} - data shape: {batch_fake.shape}', fontsize=12, ha='left', va='top', transform=ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 4})
        
    
    return figure