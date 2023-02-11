#  Authors:	Kay Hartmann <kg.hartma@gmail.com>
#  		    Samuel Böhm <samuel-boehm@web.de>

import copy
import os
from collections import OrderedDict
from typing import List, MutableMapping, Tuple
import joblib
import numpy as np
import mne
import logging
from scipy import linalg
from moabb.datasets import Schirrmeister2017
from braindecode.preprocessing import exponential_moving_standardize

from eeggan.data.preprocess.resample import upsample, downsample
from eeggan.examples.high_gamma.dataset import HighGammaDataset
from eeggan.data.dataset import SignalAndTarget
from eeggan.validation.deep4 import train_completetrials
from eeggan.data.preprocess.util import prepare_data
from eeggan.eeggan_logger import set_logger_level, init_logger

logger = logging.getLogger(__name__)
init_logger(logger, level='INFO')

def make_dataset_for_subj(name: str, dataset_path: str,
                          channels: List[str], classdict: OrderedDict,
                          fs: float, interval_times: Tuple[float, float], verbose='INFO'):
    """
    Loads subject from MOABB Dataset using Braindecode and saves a single dataset instance for the subject.

    Args:
        name (str): dataset name
        
        dataset_path (str): target path, where the fresh created dataset is dumped
        
        channels (List[str]): List of channels to extract
        
        classdict (MutableMapping[str, int]): dict of class labels
        
        fs (float): sampling rate to which the dataset is (re)-sampled

        interval_times (Tuple[float, float]): start and stop after in seconds

        verbose (str): set logger level
    """

    set_logger_level(logger, verbose)

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

    test_set = prepare_data(test_set.X, test_set.y, n_classes, train_set.X.shape[2], normalize=True)
    train_set = prepare_data(train_set.X, train_set.y, n_classes, test_set.X.shape[2], normalize=True)

    test_set.index_dict = data_collection['test'].index_dict
    train_set.index_dict = data_collection['train'].index_dict

    dataset = HighGammaDataset(train_set, test_set, train_set[0][0].shape[1],
                               channels, [e for e in classdict.values()], fs)

   
    joblib.dump(dataset, os.path.join(dataset_path, f'{name}.dataset' ), compress=True)

def make_deep4_for_subj(name: str, dataset_path: str, deep4_path: str, n_progressive: int, n_deep4: int,
                        verbose='INFO', n_epochs: int=100):

    set_logger_level(logger, level=verbose)

    if not os.path.exists(deep4_path):
        os.makedirs(deep4_path)


    dataset = load_dataset(name, dataset_path)

    logger.debug(f'Dataset contains {len(dataset.classes)} classes, {len(dataset.channels)}'
                 f' channels and was sampled in {dataset.fs} Hz')

    n_classes = len(dataset.classes)
    n_chans = len(dataset.channels)

    for i_stage in np.arange(n_progressive):
        models = []
        train_set_stage = copy.copy(dataset.train_data)
        test_set_stage = copy.copy(dataset.test_data)

        logger.info(f'make data for stage {i_stage}')

        train_set_stage.X = make_data_for_stage(train_set_stage.X, i_stage, n_progressive - 1)
        test_set_stage.X = make_data_for_stage(test_set_stage.X, i_stage, n_progressive - 1)
        logger.debug(f'trainset got downsampled from shape {dataset.train_data.X.shape} to '
                     f'shape {train_set_stage.X.shape}')

        for i_deep4 in range(n_deep4):
            deep4_dict_path = f'{deep4_path}/{name}_stage{i_stage}_{i_deep4}' 
            mod = make_deep4(train_set_stage, test_set_stage, n_classes, n_chans, deep4_dict_path, n_epochs)
            models.append(mod)
        
        print('Number of Models:', len(models))

        joblib.dump(models, os.path.join(deep4_path, f'{name}_stage{i_stage}.deep4'), compress=True)


def make_data_for_stage(X, i_stage, max_stage):
    down = downsample(X, 2 ** (max_stage - i_stage), axis=2)
    return upsample(down, 2 ** (max_stage - i_stage), axis=2)


def make_deep4(train_set, test_set, n_classes, n_chans, deep4_path, n_epochs):

    batch_size = 64

    model = train_completetrials(train_set, test_set, n_classes, n_chans, deep4_path,
                                 n_epochs=n_epochs, batch_size=batch_size, 
                                 cuda=True)
    model = model.cpu().eval()
    return model


def load_dataset(name: str, path: str) -> HighGammaDataset:
    return joblib.load(os.path.join(path, f'{name}.dataset'))


def load_deeps4(name: str, stage: int, path: str):
    return joblib.load(os.path.join(path, f'{name}_deep4_stage{stage}.deep4'))


def fetch_and_unpack_schirrmeister2017_moabb_data(channels: List[str],
                                                  interval_times: Tuple[float, float], fs: float, mapping: dict):
    # Get raw data from MOABB
    DataSet = {'test': SignalAndTarget(np.array([]), np.array([]), None),
               'train': SignalAndTarget(np.array([]), np.array([]), None) }
    mne.set_log_level('WARNING')
    data = Schirrmeister2017().get_data()
    for subj_id, subj_data in data.items():
        logger.info(f'Preprocessing Data for subj {subj_id}')
        for sess_id, sess_data in subj_data.items():
            for run_id, raw in sess_data.items():
                X, y = _preprocess_and_stack(raw, channels, interval_times, fs, mapping)
                DataSet[run_id].add_data(X, y, subj_id)
    return DataSet


def _preprocess_and_stack(raw, channels, interval_times, fs, mapping):
    n_total_chs = len(raw.info['ch_names'])
    raw = raw.pick(picks=channels)
    # Preprocess:
    raw.load_data()
    raw.set_eeg_reference('average', projection=False)
    old_fs = raw.info['sfreq']
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