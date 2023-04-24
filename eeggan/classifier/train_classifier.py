#  Authors:	Samuel Böhm <samuel-boehm@web.de>
import joblib
import os
from eeggan.data.create_dataset import ZCA_whitening
from braindecode.preprocessing import exponential_moving_standardize, create_windows_from_events

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor

dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=list(range(1, 15)))

# safe path for dataset
dataset_path = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/eeggan2.0'

# targed sfreq
sfreq = 256

# Channels to pick
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4',
            'P8', 'O1', 'O2', 'M1', 'M2']

# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

# Define Preprocessor
preprocessors = [
    Preprocessor('resample', sfreq),
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)

# Create wondowed dataset
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0.5*sfreq,
    window_size_samples=2.5*sfreq, window_stride_samples=100,
    drop_last_window=False, picks = channels)

# Safe Dataset
joblib.dump(windows_dataset, os.path.join(dataset_path, 'windowed.dataset' ), compress=True)