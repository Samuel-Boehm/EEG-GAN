#  Authors:	Samuel Böhm <samuel-boehm@web.de>

import copy
import os
from collections import OrderedDict
from typing import List, Tuple
import joblib
import numpy as np
import mne
from scipy import linalg
from moabb.datasets import Schirrmeister2017
from braindecode.preprocessing import exponential_moving_standardize

from eeggan.data.dataclasses import Dataset, Data

def make_dataset_for_subj(name: str, dataset_path: str,
                          channels: List[str], classdict: OrderedDict,
                          fs: float, interval_times: Tuple[float, float]):
    """
    Loads subject from MOABB Dataset using Braindecode and saves a single dataset instance for the subject.

    Args:
        name (str): dataset name
        
        dataset_path (str): target path, where the fresh created dataset is dumped
        
        channels (List[str]): List of channels to extract
        
        classdict (MutableMapping[str, int]): dict of class labels
        
        fs (float): sampling rate to which the dataset is (re)-sampled

        interval_times (Tuple[float, float]): start and stop after in seconds

    """

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    n_classes = len(classdict)
    data_collection = fetch_and_unpack_schirrmeister2017_moabb_data(
        channels=channels,
        interval_times=interval_times,
        fs=fs,
        mapping=classdict)

    test_set = data_collection['test']
    train_set = data_collection['train']

    test_set.prepare_data(normalize=True)
    train_set.prepare_data(normalize=True)

    dataset = Dataset(train_set, test_set)

   
    joblib.dump(dataset, os.path.join(dataset_path, f'{name}.dataset' ), compress=True)


def load_dataset(name: str, path: str) -> Dataset:
    return joblib.load(os.path.join(path, f'{name}.dataset'))


def fetch_and_unpack_schirrmeister2017_moabb_data(channels: List[str],
                                                  interval_times: Tuple[float, float], fs: float, mapping: dict):
    # Get raw data from MOABB
    # Create Dictionary with empty Data containers
    DataSet = {'test':Data(np.array(), np.array()),
               'train':Data(np.array(), np.array())}
    
    mne.set_log_level('WARNING')
    data = Schirrmeister2017().get_data()
    for subj_id, subj_data in data.items():
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                X, y = _preprocess_and_stack(raw, channels, interval_times, fs, mapping)
                DataSet[run_id].add_data(X, y, subj_id)
    return DataSet


def _preprocess_and_stack(raw, channels, interval_times, fs, mapping):
    raw = raw.pick(picks=channels)
    # Preprocess:
    raw.load_data()
    raw.set_eeg_reference('average', projection=False)
    raw.resample(fs)
    raw.apply_function(exponential_moving_standardize, channel_wise=False,
                       init_block_size=1000, factor_new=0.001, eps=1e-4)
    # Extract events (trials):
    events, events_id = mne.events_from_annotations(raw, mapping)

    start_in_seconds = interval_times[0]

    # Remove one timepoint from stop: 
    stop_in_seconds = interval_times[1] - (1/fs)

    mne_epochs = mne.Epochs(raw, events, event_id=events_id, tmin=start_in_seconds, tmax=stop_in_seconds, baseline=None)
    mne_epochs.drop_bad()

    X = mne_epochs.get_data()
    X = X.astype(dtype=np.float32)
    X = ZCA_whitening(X)
    annots = mne_epochs.get_annotations_per_epoch()
    labels = [x[0][-1] for x in annots]
    y = [mapping[k] for k in labels]
    return X, np.array(y)

def ZCA_whitening(X):
    
    '''
    Applies zero component analysis whitening to the input X
    X needs to be of shape (trials, channels, datapoints)

    if a stim onset is given (in seconds), the covariance matrix will only be calculated with 
    the data that was until the stimulus.
    '''
    X_whitened = np.zeros_like(X)

    for i in range (X.shape[0]): 
        # Zero center data
        xc = X[i] - np.mean(X[i], axis=0)

        xcov = np.cov(xc, rowvar=True, bias=True)

        # Calculate Eigenvalues and Eigenvectors
        w, v = linalg.eig(xcov) # 
        
        # Create a diagonal matrix
        diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
        diagw = diagw.real.round(4) #convert to real and round off\

        # Whitening transform using ZCA (Zero Component Analysis)

        X_whitened[i] = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
    return X_whitened