# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader

import os
import numpy as np

from pathlib import Path

import yaml

from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets import create_from_X_y
from braindecode.preprocessing import preprocess, Preprocessor

# Mute braindecode and MNE
import logging
logging.getLogger('mne').setLevel(logging.ERROR)
logging.getLogger('braindecode').setLevel(logging.ERROR)


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

    def __init__(self, dataset_name: str, batch_size: int, n_stages: int, **kwargs) -> None:

        self.debug = False
        if dataset_name == 'debug':
            print('Dataloader entering debug mode, only using dummy data.')
            self.debug = True

            path = Path.cwd() / 'configs' / 'data' / 'debug.yaml'

            if not path.exists():
                raise FileNotFoundError(f'''Could not find the file {path}. The dataloader is in debug mode and requires a
                                        debug.yaml file in the configs/data directory to generate dummy data.''')

            with open(path, 'r') as file:
                self.data_dict = yaml.safe_load(file)

        self.data_dir = Path.cwd() / 'datasets' / dataset_name
        self.batch_size = batch_size
        self.n_stages = n_stages
        super().__init__()

    def setup(self, stage: str) -> None:
        self.set_stage(1)

    def train_dataloader(self) -> DataLoader:
        dl = ThrowAwayIndexLoader(self.data, batch_size=self.batch_size,
                                  shuffle=True, num_workers=2)
        return dl

    def test_dataloader(self) -> None:
        return None

    def set_stage(self, stage: int):
        stage = self.n_stages - stage  # override external with internal stage vatiable
        if not self.debug:
            ds_list = []
            for ds in Path(self.data_dir).rglob('S*.pt'):
                ds_list.append(torch.load(ds))
            self.data = BaseConcatDataset(ds_list)

            base_sfreq = self.data.description['fs'][0]
            current_sfreq = int(base_sfreq // 2**stage)

            # If the data is already at the correct frequency, we don't need to resample it.
            if current_sfreq == base_sfreq:
                return

            preprocessors = [Preprocessor(change_type, out_type='float64', picks='all')]
            preprocessors.append(Preprocessor(
                'resample', sfreq=current_sfreq, npad=0))
            preprocessors.append(Preprocessor(change_type, out_type='float32', picks='all'))
            self.data = preprocess(self.data, preprocessors, n_jobs=-1)
            return

        else:
            time_in_seconds = self.data_dict['length_in_seconds']
            sfreq = self.data_dict['sfreq']
            n_samples_curent_stage = int(time_in_seconds * (sfreq // 2**stage))
            n_channels = len(self.data_dict['channels'])
            n_classes = len(self.data_dict['classes'])
            X = np.random.randn(4*self.batch_size,
                                n_channels, n_samples_curent_stage)
            y = np.random.randint(0, n_classes, 4*self.batch_size)

            windows_dataset = create_from_X_y(
                X, y, drop_last_window=False, sfreq=sfreq, ch_names=self.data_dict['channels'])
            self.data = BaseConcatDataset([windows_dataset])
