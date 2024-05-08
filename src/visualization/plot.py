import matplotlib.pyplot as plt
import numpy as np

def calculate_spectrum(data, sfreq):
    n_samples = data.shape[-1]
    n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
    f = np.fft.rfftfreq(n_fft, 1/sfreq)
    spectrum = np.abs(np.fft.rfft(data, n=n_fft, axis=1))
    return f, spectrum

def plot_spectrum(data, sfreq, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    f, spectrum = calculate_spectrum(data, sfreq)
    ax.plot(f, np.mean(spectrum, axis=0), label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Spectrum")
    return ax



if __name__ == "__main__":
    sfreq = 246
    n_samples = 512
    n_channels = 5
    batch_size = 32
    data = np.random.randn(batch_size, n_channels, n_samples)
    plot_spectrum(data, sfreq)
    plt.show()