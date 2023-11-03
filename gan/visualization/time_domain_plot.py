# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import matplotlib.pyplot as plt
import numpy as np
from gan.data.batch import batch_data


def plot_time_domain(batch:batch_data, channel_names:list, fs:float, path:str):
    '''
    Plots the time domain of the real and fake data.
    
    Arguments:
    ----------
        real: np.ndarray
            The real data.
        fake: np.ndarray
            The fake data.
        channel_names: list
            The names of the channels.
        path: str
            The path where the plot should be saved.
    '''
    
    fig, axs = plt.subplots(7, 3,  figsize=(8.27, 11.69), facecolor='w')

    x = np.linspace(-.5, (batch.shape()[-1] / fs) -.5, num = batch.shape()[-1])

    for i, ax in enumerate(axs.flatten()):
        if channel_names:
            ax.text(.94, .94, channel_names[i], fontsize=12, ha='right', va='top', transform=ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 4})
        
        real_mean = batch.real.mean(axis = 0)
        fake_mean = batch.fake.mean(axis = 0)

        stack = np.concatenate((real_mean, fake_mean))
        ax.plot(x, real_mean[i], c='#c1002a', label ='real', lw = 1)
        ax.plot(x, fake_mean[i], c='#004a99', label='fake', lw = 1)
        
        max_y = stack.max() + stack.max() * .1
        min_y = stack.min() + stack.min() * .1

        ax.set_ylim(bottom=min_y, top=max_y)
        # ax.fill_between(x, fake.mean(axis = 0)[i] + fake.std(axis = 0)[i], fake.mean(axis = 0)[i] - fake.std(axis = 0)[i], color='#004a99', alpha=.4)
        # ax.fill_between(x, real.mean(axis = 0)[i] + real.std(axis = 0)[i], real.mean(axis = 0)[i] - real.std(axis = 0)[i], color='#c1002a', alpha=.4)

        if i % 3 > 0:
            ax.set_yticks([])
        
        else:
            if i == 10:
                ax.set_ylabel('Amplitude', fontsize=12)

        if i // 3 != 6:
            ax.set_xticks([])
        else:
            if i == 17:
                ax.set_xlabel('time [s]', fontsize=12)

    #plt.tight_layout()

    plt.legend(loc=4)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if path:
        plt.savefig(path, format='pdf', bbox_inches='tight')
        
    else:
        return fig, axs