# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
from numpy import multiply, clip
import numpy.linalg as linalg

import hydra
from omegaconf import DictConfig

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from braindecode.datautil.preprocess import exponential_moving_standardize



def ZCA_whitening(data:np.ndarray) -> np.ndarray:
    r"""
    Perform zero component analysis whitening.
   
    Parameters:
    ----------
    data (np.ndarray): input data (trials, channels, datapoints)
    
    Returns
    -------
    whitened data
    """
    
    # Zero center data
    xc = data - np.mean(data, axis=0)
    xcov = np.cov(xc, rowvar=True, bias=True)

    # Calculate Eigenvalues and Eigenvectors
    w, v = linalg.eig(xcov) 
    
    # Create a diagonal matrix
    diagw = np.diag(1/(w**0.5)) 
    diagw = diagw.real.round(4) 

    # Whitening transform using ZCA (Zero Component Analysis)
    return np.dot(np.dot(np.dot(v, diagw), v.T), xc)


@hydra.main(config_path="configs", config_name="data_config")
def preprocess(cfg: DictConfig) -> None:

    dataset = MOABBDataset(dataset_name=cfg.dataset_name, subject_ids=cfg.subject_ids, preload=True)

    low_cut_hz = cfg.low_cut  # low cut frequency for filtering
    high_cut_hz = cfg.high_cut  # high cut frequency for filtering
    
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    
    # Factor to convert from V to uV
    factor = 1e6

    preprocessors = []

    preprocessors.append(Preprocessor('pick', picks=cfg.channels))
    preprocessors.append(Preprocessor('pick_types', eeg=True, meg=False, stim=False))
    preprocessors.append(Preprocessor('set_eeg_reference', ref_channels='average'))
    if hasattr(cfg, 'ZCA_whitening'):
        preprocessors.append(Preprocessor(ZCA_whitening))
    preprocessors.append(Preprocessor(lambda data: multiply(data, factor)))
    preprocessors.append(Preprocessor(lambda data: clip(data, a_min=-800., a_max=800.), channel_wise=True))
    preprocessors.append(Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz))
    if hasattr(cfg, 'sfreq'):
        preprocessors.append(Preprocessor('resample', sfreq=cfg.sfreq))
    preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size))
    
    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=-1)


    total_samples = 3.5 * cfg.sfreq # This is hardcoded here for Schirrmeister2017 dataset... how to draw this from the dataset?

    out_length = 2.5
    duration_before_trial_start = 0.5

    trial_length_samples = int(out_length * sfreq)

    trial_start_offset_samples = -int(duration_before_trial_start * sfreq)
    trial_stop_offset_samples = -int(total_samples - trial_start_offset_samples - trial_length_samples)

    print(trial_start_offset_samples, trial_stop_offset_samples)

    trial_start_offset_samples = int(-0.5 * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        preload=True,
)



