# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

# This file is used to create a set of dummy data, this can be used to test model behavior.

import numpy as np

def generate_tone_burst(frequency, sampling_rate, duration, signal_duration, shift_percent=0, noise_level=0.1):
    # Calculate the number of samples for the tone burst and the final signal
    n_samples_burst = int(sampling_rate * duration)
    n_samples_signal = int(sampling_rate * signal_duration)
    
    # Generate tone burst
    t_burst = np.arange(n_samples_burst) / sampling_rate
    sine_wave = np.sin(frequency * t_burst)
    envelope = np.hanning(n_samples_burst)
    tone_burst = sine_wave * envelope
    
    # Create the final signal with zeros
    final_signal = np.zeros(n_samples_signal)
    
    # Determine the position for the tone burst based on shift_percent
    if shift_percent <= 50:
        # Tone burst starts at this percentage
        start_index = int(n_samples_signal * (shift_percent / 100))
    else:
        # Tone burst ends at this percentage
        end_percentage = shift_percent / 100
        start_index = int(n_samples_signal * end_percentage) - n_samples_burst
    
    end_index = start_index + n_samples_burst
    
    # Ensure the tone burst fits within the final signal duration
    if 0 <= start_index and end_index <= n_samples_signal:
        # Insert the tone burst into the final signal at the calculated position
        final_signal[start_index:end_index] = tone_burst
    else:
        raise ValueError("Tone burst with the specified shift does not fit within the signal duration.")
    
    # Add noise to the final signal
    noise = np.random.normal(0, noise_level, n_samples_signal)
    final_signal_with_noise = final_signal + noise
    
    # Time vector for the final signal
    t_signal = np.arange(n_samples_signal) / sampling_rate
    
    return t_signal, final_signal_with_noise


def create_dummy_data(n_samples:int, n_channels:int, n_classes:int,
                      sfreq:int, length_seconds:float) -> np.ndarray:
    '''
    Create a set of dummy data.

    Parameters:
    ----------
    n_samples (int): number of samples to create per class
    n_channels (int): number of channels to create
    n_classes (int): number of classes to create
    sfreq (int): sampling frequency of the data
    length_seconds (float): length of the data in seconds

    Returns:
    ----------
    np.ndarray: the created data
    '''
        
    # for each class we n_channels channels, each channel should have a different amplitude
    # each class should have a different frequency
    
    frequencies = np.linspace(10, (sfreq // 2) -10, n_channels)

    for cls in range(n_classes):
        data = np.zeros((n_samples, n_channels, int(sfreq*length_seconds)))
        for i in range(n_samples):
            for j in range(n_channels):
                # Example usage
                # We want the frequency to increase with the channel number relative to the sampling frequency
                frequency = frequencies[j]
                duration = 0.5  # Duration of the tone burst in seconds
                if cls == 0:
                    shift_percent = ((j / n_channels) * 100) # Shift the tone burst relative to the signal duration in percent
                else:
                    shift_percent = ((n_channels - j) / n_channels * 100)
                noise_level = 0.02  # Noise level

                _, data[i, j] = generate_tone_burst(frequency, sfreq, duration, length_seconds, shift_percent, noise_level)
               
        if cls == 0:
            X = data
            y = np.ones(n_samples) * cls
        else:    
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.ones(n_samples) * cls))

        # Normalize Amplitude
        X = X / np.max(X)
    
    
    return X, y

if __name__ == '__main__':
    from braindecode.datasets import create_from_X_y
    from torch import save
    from pathlib import Path

    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = Path(base_dir, "datasets", "dummy_data")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    SFREQ = 256

    N_CHANNELS = 21
    CH_NAMES = [f'ch_{i}' for i in range(N_CHANNELS)]

    X, y = create_dummy_data(800, N_CHANNELS, 2, SFREQ, 2.5)

    dummy_ds = create_from_X_y(X, y, drop_last_window=False, sfreq=SFREQ, ch_names=CH_NAMES,)
    
    dummy_ds.set_description({'fs': [SFREQ] * len(dummy_ds.datasets)})

    dataset_path = Path(dataset_dir, f"S1.pt")
    with open(dataset_path, 'wb') as f:
        save(dummy_ds, f)
