import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def calculate_spectrum(data:np.ndarray, sfreq) -> Tuple[np.ndarray, np.ndarray]:
    # Check dimensions
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]
    if len(data.shape) != 3:
        raise ValueError(f"Data must have shape (batch_size, n_channels, n_samples) or (n_channels, n_samples), got {data.shape}")
    
    n_samples = data.shape[-1]
    n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
    f = np.fft.rfftfreq(n_fft, 1/sfreq)
    spectrum = np.mean(np.abs(np.fft.rfft(data, n=n_fft, axis=-1)), axis=0)
    return f, spectrum

def plot_spectrum(data, sfreq, ax=None, title=None, label=None, show_std=False):
    if ax is None:
        fig, ax = plt.subplots()
    f, spectrum = calculate_spectrum(data, sfreq)
    ax.plot(f, np.mean(spectrum, axis=0), label=label)
    if show_std:
        ax.fill_between(f, np.mean(spectrum, axis=0) - np.std(spectrum, axis=0), np.mean(spectrum, axis=0) + np.std(spectrum, axis=0), alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    return ax

def plot_time_domain(data, ax=None, title=None, label=None, show_std=False):
    
    if len(data.shape) != 2:
        raise ValueError(f"Data must have shape (batch_size, n_samples) got {data.shape}")
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.mean(data, axis=0).T, label=label)
    if show_std:
        ax.fill_between(np.arange(data.shape[1]), np.mean(data, axis=0).T - np.std(data, axis=0).T, np.mean(data, axis=0).T + np.std(data, axis=0).T, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    return ax

def split_into_labels(data:np.ndarray, targets:np.ndarray, mapping:dict=None):
    unique_labels = np.unique(targets)
    # Check if mapping keys are in unique_labels
    if mapping is not None:
        for key in mapping.keys():
            if key not in unique_labels:
                raise ValueError(f"Mapping key {key} not found in labels")
    
    data_dict = {}
    for label in unique_labels:
        if mapping is not None:
            label_name = mapping[label]
        else:
            label_name = label
        data_dict[label_name] = data[targets == label]
    return data_dict

def plot_time_domain_by_target(data, targets, ax=None, show_std=False, title=None, mapping=None) -> Tuple[plt.Axes, plt.Figure]:
    fig = None
    data_dict = split_into_labels(data, targets, mapping)
    if ax is None:
        fig, ax = plt.subplots()
    for target, data in data_dict.items():
        plot_time_domain(data, ax=ax, title=title, show_std=show_std, label=target)
    ax.legend()
    return ax, fig

def plot_spectrum_by_target(data, targets, sfreq, ax=None, show_std=False, title=None, mapping=None) -> Tuple[plt.Axes, plt.Figure]:
    fig = None
    data_dict = split_into_labels(data, targets, mapping)
    if ax is None:
        fig, ax = plt.subplots()
    for target, data in data_dict.items():
        plot_spectrum(data, sfreq, ax=ax, title=title, show_std=show_std, label=target)
    ax.legend()
    return ax, fig

if __name__ == "__main__":
    sfreq = 256
    n_samples = 512 
    n_channels = 6
    batch_size = 32
    # generate 5 channels of different sine
    data = np.zeros((batch_size, n_channels, n_samples))

    for i in range(n_channels):
        data[:, i, :] = np.sin(np.linspace(0, 2*np.pi, n_samples) * (i+1) * 20 ) + np.random.randn(batch_size, n_samples) * .3
    
    labels = np.random.randint(0, 2, batch_size)
    
    ax, fig = plot_time_domain_by_target(data[:, 3, :], labels, show_std=True, mapping={0:"Class 0", 1:"Class 1"})
    ax, fig = plot_spectrum_by_target(data[:, 3, :], labels, sfreq, show_std=True, mapping={0:"Class 0", 1:"Class 1"})

    plt.show()