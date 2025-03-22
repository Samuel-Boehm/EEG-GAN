# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
from numpy import multiply, clip

import torch
import mne 
import numpy.linalg as linalg


from pathlib import Path
from joblib import Parallel, delayed

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from braindecode.datautil.preprocess import exponential_moving_standardize

def get_dataset_info(dataset_name: str) -> tuple:
    """
    Helper function that returns the interval, event_id and subject_list for a given dataset.
    """
    from moabb.datasets.utils import dataset_list
    for dataset in dataset_list:
        if dataset_name == dataset.__name__:
            return dataset().event_id, dataset().subject_list
    raise ValueError(f"{dataset_name} not found in moabb datasets")

def extract_data(
    dataset: MOABBDataset,
    preprocessors: list,
    tmin: float,
    tmax: float,
    event_map: dict,
    sfreq: float ) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Applies preprocessing to the dataset and extracts the data, labels and metadata.
    """

    X, y, split = [], [], []
    preprocess(dataset, preprocessors)
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    
    n_samples = int((tmax - tmin) * dataset.datasets[0].raw.info['sfreq'])
    
    for i, ds in enumerate(dataset.datasets):
        events, event_ids = mne.events_from_annotations(ds.raw, event_id=event_map)
        epochs = mne.Epochs(ds.raw, events, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=None, event_repeated='drop')

        X_ = epochs.get_data()[:, :, :-1]
        y_ = epochs.events[:, 2]
        assert X_.shape[-1] == n_samples, f"Expected {n_samples} samples, got {X_.shape[-1]}. Total shape: {X_.shape}"

        X.append(X_)
        y.append(y_)

        split.append([ds.description['run'][1:]] * len(y_))

    X = torch.tensor(np.concatenate(X))
    y = torch.tensor(np.concatenate(y))
    split = np.concatenate(split, dtype=object)

    return X, y, split

def normalize(data:np.ndarray):
    """
    Normalize data to zero mean and unit variance
    """
    return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)


def  _2D_ZCA_whitening(data:np.ndarray, trial_start_offset_samples:int) -> np.ndarray:
    
    # Zero center data
    xc = data - np.mean(data, axis=1, keepdims=True)

    # Calculate Covariance Matrix only on datapoints before the trial start
    xcov = np.cov(xc[:, :np.abs(trial_start_offset_samples)], rowvar=True, bias=True)

    # Calculate Eigenvalues and Eigenvectors
    w, v = linalg.eig(xcov) 

    # Create a diagonal matrix
    diagw = np.diag(1/(w**0.5)) 
    diagw = diagw.real.round(4) 

    # Whitening transform using ZCA (Zero Component Analysis)
    return np.dot(np.dot(np.dot(v, diagw), v.T), xc)

def apply_zca_whitening(data:np.ndarray, trial_start_offset_samples:int) -> np.ndarray:
    r"""
    Perform zero component analysis whitening.
   
    Parameters:
    ----------
    data (np.ndarray): input data (trials, channels, datapoints)
    trial_start_offset_samples (int): offset of the trial start in samples

    Returns
    -------
    whitened data (np.ndarray)
    """

    data_whitened = Parallel(n_jobs=-1)(delayed(_2D_ZCA_whitening)
                                        (data[i], trial_start_offset_samples)
                                        for i in range(data.shape[0]))
    
    return np.stack(data_whitened, dtype=np.float32)

def preprocess_moabb(dataset_name: str, dataset_dir:str, channels: list, tmin: float, tmax:float, sfreq:float,
                    classes:list, ZCA_whitening: bool=False, normalize:bool=False, *args, **kwargs) -> None:

    _, subject_list = get_dataset_info(dataset_name)

    # create mapping from event to class
    label_mapping = {event: i for i, event in enumerate(classes)}

    for subject in subject_list:
           
        dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subject])
        
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        
        # Factor to convert from V to uV
        factor = 1e6

        # First we apply global preprocessing steps
        preprocessors = []

        preprocessors.append(Preprocessor('pick', picks=channels))
        preprocessors.append(Preprocessor('pick_types', eeg=True, meg=False, stim=False))
        preprocessors.append(Preprocessor('set_eeg_reference', ref_channels='average'))
        preprocessors.append(Preprocessor(lambda data: multiply(data, factor)))
        preprocessors.append(Preprocessor(lambda data: clip(data, a_min=-800., a_max=800.), channel_wise=True))
        preprocessors.append(Preprocessor('resample', sfreq=sfreq, npad=0))  
        preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size))
        
        # Extract data
        X, y, split = extract_data(dataset, preprocessors, tmin, tmax, label_mapping, sfreq)

        # Now sample wise preprocessing
        
        if ZCA_whitening:
            trial_start_offset_samples = int(0.5 * sfreq)
            X_np = X.numpy()
            X_np = apply_zca_whitening(X_np, trial_start_offset_samples=trial_start_offset_samples)
            X = torch.tensor(X_np)
            
        if normalize:
            # Convert X to numpy array, apply normalization, then convert back to tensor
            X_np = X.numpy()
            X_np = normalize(X_np)
            X = torch.tensor(X_np)
        
        # Save data
        subject_data = {"X": X, "y": y, "split": split}
        subject_path = Path(dataset_dir, f"S{subject}.pt")
        torch.save(subject_data, subject_path)

    return label_mapping




