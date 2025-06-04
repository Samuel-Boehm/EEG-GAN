import numpy as np


def single_event(
    n_seconds,
    sfreq,
    num_channels=21,
    n_trials=50,
    differing_channels=[3, 7, 8, 9, 18],
    event_amplitude=1,  # Amplitude of the event
    noise_amplitude=5,  # Amplitude of the noise
):
    """
    Generate fake EEG data with two classes.

    Parameters:
    ----------
    n_seconds : float
        Duration of each trial in seconds.
    sfreq : int
        Sampling frequency in Hz.
    num_channels : int, optional
        Number of EEG channels. Defaults to 21.
    num_trials_per_class : int, optional
        Number of trials for each class. Defaults to 50.
    differing_channels : list of int, optional
        List of channel indices where the event will differ between classes.
        Defaults to [3, 7, 8, 9, 18].
    event_amplitude : float, optional
        Amplitude of the event waveform. Defaults to 5.
    noise_amplitude : float, optional
        Standard deviation of the noise. Defaults to 100.

    Returns:
    --------
    X : ndarray
        EEG data with shape (num_trials, num_channels, num_samples)
    y : ndarray
        Class labels (0 or 1)
    """
    num_samples = int(n_seconds * sfreq)

    y = np.ones(n_trials, dtype=int)
    y[:n_trials] = 0  # First half are class 0, second half are class 1

    # Create a EEG-like event:
    event_duration = 0.2  # seconds
    event_start_time = 0.5  # seconds into the trial
    event_samples = int(event_duration * sfreq)
    event_start_idx = int(event_start_time * sfreq)

    # Ensure event fits within the trial
    if event_start_idx + event_samples > num_samples:
        event_samples = num_samples - event_start_idx
        if event_samples <= 0:
            raise ValueError(
                "Event start time is too late for the given n_seconds and event_duration."
            )
        print(
            f"Warning: Event duration adjusted to {event_samples / sfreq:.2f}s to fit within trial."
        )

    # Time vector for the event waveform
    event_time_vector = np.linspace(0, event_duration, event_samples, endpoint=False)

    # Define Event Waveforms for Each Class
    # Event for Class 0: A positive sine wave pulse
    event_class0_waveform = event_amplitude * np.sin(
        2 * np.pi * 10 * event_time_vector
    )  # 10 Hz sine wave

    # Event for Class 1: A negative sine wave pulse (inverted phase or frequency)
    event_class1_waveform = -event_amplitude * np.sin(
        2 * np.pi * 10 * event_time_vector
    )  # 10 Hz inverted sine wave

    # Generate Noise (this will be added to all channels and trials)
    overall_noise = np.random.normal(
        0, noise_amplitude, size=(n_trials, num_channels, num_samples)
    )

    # Add initial noise to X
    X = overall_noise.copy()  # Start with pure noise

    # Insert Events into Specific Channels
    for trial_idx in range(n_trials):
        current_class = y[trial_idx]

        for channel_idx in range(num_channels):
            if channel_idx in differing_channels and current_class == 0:
                # Add event for Class 0
                X[
                    trial_idx,
                    channel_idx,
                    event_start_idx : event_start_idx + event_samples,
                ] += event_class0_waveform
            else:
                # Add event for Class 1
                X[
                    trial_idx,
                    channel_idx,
                    event_start_idx : event_start_idx + event_samples,
                ] += event_class1_waveform
    return X, y


def sines(n_seconds, sfreq, num_channels, n_trials, differing_channels=None):
    X = np.zeros((n_trials, num_channels, int(n_seconds * sfreq)))
    y = np.zeros((n_trials,), dtype=int)
    time = np.arange(n_seconds * sfreq) / sfreq
    for i in range(num_channels):
        signal = np.sin(2 * np.pi * 0.5 * time)
        X[:, i, :] = signal

    # Add noise
    noise = np.random.normal(0, 0.5, X.shape)
    X += noise
    if differing_channels is not None:
        for ch in differing_channels:
            X[int(n_trials / 2) :, ch, :] = np.sin(2 * np.pi * 2 * time + np.pi / 4)
    y[: int(n_trials / 2)] = 1
    shuffled_indices = np.random.permutation(n_trials)
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    return X, y


def individual_sines(n_seconds, sfreq, num_channels, n_trials, differing_channels=None):
    X = np.zeros((n_trials, num_channels, int(n_seconds * sfreq)))
    y = np.zeros((n_trials,), dtype=int)
    freqs = np.linspace(0.25, 64, num_channels)
    time = np.arange(n_seconds * sfreq) / sfreq
    for i in range(num_channels):
        signal = np.sin(2 * np.pi * freqs[i] * time)
        X[:, i, :] = signal
    # Add noise
    noise = np.random.normal(0, 0.01, X.shape)
    X += noise
    if differing_channels is not None:
        for ch in differing_channels:
            X[int(n_trials / 2) :, ch, :] = np.sin(2 * np.pi * 2 * time + np.pi / 4)
    y[: int(n_trials / 2)] = 1
    shuffled_indices = np.random.permutation(n_trials)
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    return X, y
