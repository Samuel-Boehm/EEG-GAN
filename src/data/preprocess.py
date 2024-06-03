# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
from numpy import multiply, clip
import numpy.linalg as linalg

import hydra
import omegaconf
from omegaconf import DictConfig
from pathlib import Path
from joblib import Parallel, delayed

import torch
import pandas as pd

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
from braindecode.datautil.preprocess import exponential_moving_standardize

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

def ZCA_whitening(data:np.ndarray, trial_start_offset_samples:int) -> np.ndarray:
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

@hydra.main(config_path="../../configs", config_name="train")
def preprocess_moabb(cfg: DictConfig) -> None:
    cfg = cfg.data
    base_dir = Path(__file__).parent.parent.parent
    dataset_dir = Path(base_dir, "datasets", cfg.dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for subject in cfg.subject_id:
           
        dataset = MOABBDataset(dataset_name=cfg.moabb_name, subject_ids=[subject])
        
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        
        # Factor to convert from V to uV
        factor = 1e6

        # First we apply global preprocessing steps
        preprocessors = []

        preprocessors.append(Preprocessor('pick', picks=cfg.channels))
        preprocessors.append(Preprocessor('pick_types', eeg=True, meg=False, stim=False))
        preprocessors.append(Preprocessor('set_eeg_reference', ref_channels='average'))
        preprocessors.append(Preprocessor(lambda data: multiply(data, factor)))
        preprocessors.append(Preprocessor(lambda data: clip(data, a_min=-800., a_max=800.), channel_wise=True))
        if hasattr(cfg, 'sfreq'):
            preprocessors.append(Preprocessor('resample', sfreq=cfg.sfreq, npad=0))
        preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size))
        
        # Transform the data
        preprocess(dataset, preprocessors, n_jobs=-1)

        # Create windows from events
        total_samples = dataset.datasets[0].raw.annotations[0]['duration'] * cfg.sfreq
        out_length = cfg.length_in_seconds
        trial_length_samples = int(out_length * cfg.sfreq)
        trial_start_offset_samples = -int(cfg.trial_start_offset_seconds * cfg.sfreq)
        trial_stop_offset_samples = -int(total_samples - trial_length_samples - trial_start_offset_samples)

        numerical_label = np.arange(len(cfg.classes))
        mapping = dict(zip(cfg.classes, numerical_label))

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            mapping=mapping,
            preload=True,
            drop_bad_windows=True,
            verbose=False)
        
        # Now sample wise preprocessing
        preprocessors = []
        
        if hasattr(cfg, 'ZCA_whitening') and cfg.ZCA_whitening:
            preprocessors.append(Preprocessor(ZCA_whitening, trial_start_offset_samples=trial_start_offset_samples))
        if hasattr(cfg, 'normalized') and cfg.normalized:
            preprocessors.append(Preprocessor(normalize))
        
        if len(preprocessors) > 0:
            preprocess(windows_dataset, preprocessors, n_jobs=-1)

        # Not the most elegant way to add the sampling frequency to the description  but it works
        windows_dataset.set_description({'fs': [256] * len(windows_dataset.datasets)})    

        dataset_path = Path(dataset_dir, f"S{subject}.pt")
        with open(dataset_path, 'wb') as f:
            torch.save(windows_dataset, f)
        

    config_path = Path(dataset_dir, "config.yaml")
    with open(config_path, 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    preprocess_moabb()



