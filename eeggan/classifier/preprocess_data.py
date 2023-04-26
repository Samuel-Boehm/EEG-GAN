#  Authors:	Samuel Böhm <samuel-boehm@web.de>
import sys
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

import numpy as np
from scipy import linalg

from braindecode.preprocessing import (exponential_moving_standardize,
                                        create_windows_from_events,
                                        preprocess, Preprocessor)

from braindecode.datasets import HGD

def ZCA_whitening(X):
    
    '''
    Applies zero component analysis whitening to the input X
    X needs to be of shape (trials, channels, datapoints)
    '''
    print(X.shape)
    
    # Zero center data
    xc = X - np.mean(X, axis=0)

    xcov = np.cov(xc, rowvar=True, bias=True)

    # Calculate Eigenvalues and Eigenvectors
    w, v = linalg.eig(xcov) # 
    
    # Create a diagonal matrix
    diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
    diagw = diagw.real.round(4) #convert to real and round off\

    # Whitening transform using ZCA (Zero Component Analysis)

    X_whitened = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
    return X_whitened


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

dataset = HGD(subject_ids=list(range(1, 15)))

# Define Preprocessor

preprocessors = [
    Preprocessor('set_eeg_reference', ref_channels='average', projection=False),
    Preprocessor('resample', sfreq=sfreq),
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size),
    # Preprocessor(ZCA_whitening, apply_on_array=True, channel_wise=False)
]

# Transform the data
preprocess(dataset, preprocessors)

print('finised preprocessor')

# Create wondowed dataset
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=int(0.5*sfreq),
    window_size_samples=int(2.5*sfreq), window_stride_samples=100,
    drop_last_window=False, picks=channels)

windows_dataset.save(path=dataset_path, overwrite=True,)


