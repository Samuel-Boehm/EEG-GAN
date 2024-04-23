# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader

import numpy as np

from pathlib import Path

from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor

from scipy.signal import resample

def change_type(X: np.ndarray, out_type: str) -> np.ndarray:
    # MNE expects the data to be of type float64. This helper function changes the type of the input data to float64.
    if out_type == 'float64':
        return X.astype('float64')
    elif out_type == 'float32':
        return X.astype('float32')
    else:
        raise ValueError(f"Unknown type {out_type}")

class ThrowAwayIndexLoader(DataLoader):
    '''
    The BaseConcatDataset returns a tuple of (X, y, i) where i is the index of the trial.
    This loader only returns the X and y.
    ''' 
    def __init__(self, data, *args, **kwargs):
        self.dataset = data
        super().__init__(data, *args, **kwargs)
        
    
    def __iter__(self):
        for X, y, _ in super().__iter__():
            yield X, y
    
    def __len__(self):
        return len(self.dataset)


class ProgressiveGrowingDataset(LightningDataModule):
    '''
    This class is a LightningDataModule that is used to train the GAN in a progressive growing manner.
    It loads all datasets from the data_dir and resamples them to the correct frequency for the current stage.

    Parameters:
    ----------
    data_dir (str): path to the directory containing the datasets
    batch_size (int): batch size for the DataLoader

    Methods:
    ----------
    setup(): loads all datasets from the data_dir
    train_dataloader(): returns a DataLoader for the training data
    set_stage(stage: int): reloads the data and resamples it to the correct frequency for the current stage
    '''

    def __init__(self, dataset_name:str, batch_size:int, n_stages:int, **kwargs) -> None:
        self.data_dir = Path.cwd() / 'datasets' / dataset_name
        self.batch_size = batch_size
        self.n_stages = n_stages
        self.set_stage(1)
        super().__init__()

    def setup(self, stage:str) -> None:
        self.set_stage(0)
        
    def train_dataloader(self) -> DataLoader:
        return ThrowAwayIndexLoader(self.data, batch_size=self.batch_size, shuffle=True)
     
    def test_dataloader(self):
        return super().test_dataloader()
    
    def set_stage(self, stage: int):
        stage = self.n_stages - stage
        ds_list = []
        for ds in Path(self.data_dir).rglob('S*.pt'):
            ds_list.append(torch.load(ds))  
        self.data = BaseConcatDataset(ds_list)

        base_sfreq = self.data.description['fs'][0]
        current_sfreq = int(base_sfreq // 2**stage)
        time_in_seconds = self.data.datasets[0][0][0].shape[-1] / base_sfreq
        n_samples_curent_stage = int(current_sfreq * time_in_seconds)

        # If the data is already at the correct frequency, we don't need to resample it.
        if current_sfreq == base_sfreq:
            return
        
        #preprocessors = [Preprocessor(change_type, out_type='float64')]
        preprocessors = [Preprocessor(resample, num=n_samples_curent_stage, axis=-1)]
        # preprocessors.append(Preprocessor(change_type, out_type='float32'))
        
        # TODO: What the fuck? 
        self.data = preprocess(self.data, preprocessors, n_jobs=-1)

        print(self.data.datasets[0][0][0].shape)

