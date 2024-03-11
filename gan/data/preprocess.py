# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import numpy.linalg as linalg
import mne

from moabb.datasets import Schirrmeister2017
from braindecode.preprocessing.preprocess import exponential_moving_standardize
from tqdm import tqdm
from gan.data.dataset import eegganDataset

def ZCA_whitening(X:np.ndarray):
    '''
    Applies zero component analysis whitening to the input X
   
    Args:
        X (np.ndarray): input data (trials, channels, datapoints)
    
    Returns: whitened X
    '''
    
    # Zero center data
    xc = X - np.mean(X, axis=0)
    xcov = np.cov(xc, rowvar=True, bias=True)

    # Calculate Eigenvalues and Eigenvectors
    w, v = linalg.eig(xcov) 
    
    # Create a diagonal matrix
    diagw = np.diag(1/(w**0.5)) 
    diagw = diagw.real.round(4) 

    # Whitening transform using ZCA (Zero Component Analysis)
    return np.dot(np.dot(np.dot(v, diagw), v.T), xc)

def _preprocess_and_stack(raw: mne.io.Raw, channels:list, interval_times:tuple,
                          fs:int, mapping:dict):
    """
    Preprocess and stack raw data from MOABB dataset.
    Args:
        raw (mne.io.Raw): Raw data from MOABB dataset.
        channels (list): List of channels to use.
        interval_times (tuple): Interval times to use.
        fs (int): Sampling frequency to use.
        mapping (dict): Event mapping to use.
    Returns:
        X (np.ndarray): Preprocessed and stacked data.
        y (np.ndarray): Labels.
    """
    # Preprocess:
    raw = raw.pick(picks=channels)
    raw.load_data()
    raw.set_eeg_reference('average', projection=False)
    raw.apply_function(np.clip, channel_wise=False, a_min=-800., a_max=800.)
    raw.resample(fs)
    raw.apply_function(exponential_moving_standardize, channel_wise=False,
                       init_block_size=1000, factor_new=0.001, eps=1e-4)
    
    # Extract events (trials):
    events, events_id = mne.events_from_annotations(raw, mapping)
    start_in_seconds = interval_times[0]
    # Remove one timepoint from stop: 
    stop_in_seconds = interval_times[1] - (1/fs)
    mne_epochs = mne.Epochs(raw, events, event_id=events_id, tmin=start_in_seconds,
                            tmax=stop_in_seconds, baseline=None)
    mne_epochs.drop_bad()
    
    # Get labels as integers:
    annots = mne_epochs.get_annotations_per_epoch()
    labels = [x[0][-1] for x in annots]
    y = [mapping[k] for k in labels]
    
    X = mne_epochs.get_data()
    X = X.astype(dtype=np.float32)
    X = np.multiply(X, 1e6)
    # X = ZCA_whitening(X)
    # Normalize:
    X = X - X.mean()
    X = X / X.std()

    return X, np.array(y)


def fetch_and_unpack_schirrmeister2017_moabb_data(channels: list,
                                                  interval_times: tuple,
                                                  fs: float,
                                                  mapping: dict):
    """
    Load and preprocess the Schirrmeister2017 MOABB dataset.
    
    Args:
        channels (list): List of channels to use.
        interval_times (tuple): Tuple of start and stop time of the interval to use.
        fs (float): Sampling frequency.
        mapping (dict): Dictionary mapping classes to integers.
    
    Returns:

    """
    # Get raw data from MOABB
    ds = eegganDataset(['subject', 'session', 'split'], interval_times, fs, mapping, channels)
    mne.set_log_level('WARNING')
    data = Schirrmeister2017().get_data()
    for subj_id, subj_data in tqdm(data.items()):
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                X, y = _preprocess_and_stack(raw, channels, interval_times, fs, mapping)
                ds.add_data(X, y, [subj_id, sess_id, run_id])
                
    return ds






