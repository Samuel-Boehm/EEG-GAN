# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import yaml

import numpy as np
from scipy import signal

from pathlib import Path

import yaml




class ProgressiveGrowingDataset(LightningDataModule):
    '''
    This class is a LightningDataModule that is used to train the GAN in a progressive growing manner.
    It loads all datasets from the data_dir and resamples them to the correct frequency for the current stage.

    Methods:
    ----------
    setup(): loads all datasets from the data_dir
    train_dataloader(): returns a DataLoader for the training data
    set_stage(stage: int): reloads the data and resamples it to the correct frequency for the current stage
    '''

    def __init__(self, folder_name:str, batch_size:int, n_stages:int, sfreq:int, **kwargs) -> None:

        self.debug = False
        if folder_name == 'debug':
            print('Dataloader entering debug mode, only using dummy data.')
            self.debug = True

            path = Path.cwd() / 'configs' / 'data' / 'debug.yaml'

            if not path.exists():
                raise FileNotFoundError(f'''Could not find the file {path}. The dataloader is in debug mode and requires a
                                        debug.yaml file in the configs/data directory to generate dummy data.''')

            with open(path, 'r') as file:
                self.data_dict = yaml.safe_load(file)

        self.data_dir = Path.cwd() / 'datasets' / folder_name
        self.batch_size = batch_size
        self.n_stages = n_stages
        self.base_sfreq = sfreq
        super().__init__()

    def setup(self, stage: str) -> None:
        self.set_stage(1)
        # Load metadata 
        self.metadata = yaml.safe_load(open(self.data_dir / 'config.yaml', 'r'))

    def train_dataloader(self) -> DataLoader:
        ds = TensorDataset(self.X[self.split == 'train'], self.y[self.split == 'train'])
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        return dl

    def val_dataloader(self) -> None:
        ds = TensorDataset(self.X[self.split == 'test'], self.y[self.split == 'test'])
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        return dl

    def set_stage(self, stage: int):
        stage = self.n_stages - stage  # override external with internal stage vatiable
        if not self.debug:
            self.X = []
            self.y = []
            self.split = []
            for ds in Path(self.data_dir).rglob('S*.pt'):
                data_dict = torch.load(ds)

                self.X.append(data_dict['X'])
                self.y.append(data_dict['y'])
                self.split.append(data_dict['split'])
            
            self.X = torch.cat(self.X)
            self.y = torch.cat(self.y)
            self.split = np.concatenate(self.split)
            
            current_sfreq = int(self.base_sfreq // 2**stage)

            # If the data is already at the correct frequency, we don't need to resample it.
            if current_sfreq == self.base_sfreq:
                return

            # Resample the data
            self.X = self.resample(self.X.numpy(), self.base_sfreq, current_sfreq)
            self.X = torch.tensor(self.X)

        else:
            time_in_seconds = self.data_dict['length_in_seconds']
            sfreq = self.data_dict['sfreq']
            n_samples_curent_stage = int(time_in_seconds * (sfreq // 2**stage))
            n_channels = len(self.data_dict['channels'])
            n_classes = len(self.data_dict['classes'])
            X = np.random.randn(4*self.batch_size,
                                n_channels, n_samples_curent_stage)
            y = np.random.randint(0, n_classes, 4*self.batch_size)


    def resample(self,
            x: np.ndarray,
             old_sfreq: float,
             new_sfreq: float,
             axis: int = -1,
             npad: int = 100,
             pad_mode: str = "reflect",
             window: str = "boxcar") -> np.ndarray:

        # Determine target length for the original, unpadded signal
        orig_len = x.shape[axis]
        target_len = int(round(orig_len * new_sfreq / old_sfreq))
        
        # Pad along the resampling axis if requested
        if npad > 0:
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (npad, npad)
            x = np.pad(x, pad_width, mode=pad_mode)
        
        # The new length for the padded signal:
        padded_len = x.shape[axis]
        new_padded_len = int(round(padded_len * new_sfreq / old_sfreq))
        
        # Optionally apply a window to taper the padded edges
        if window != "boxcar":
            win = signal.get_window(window, padded_len)
            # Reshape the window so it can be broadcast along the correct axis
            shape = [1] * x.ndim
            shape[axis] = padded_len
            win = win.reshape(shape)
            x = x * win

        # Resample the padded signal
        x_resampled = signal.resample(x, new_padded_len, axis=axis)
        
        # Remove the padded segments from the resampled data.
        # Compute the resampled padding length:
        new_npad = int(round(npad * new_sfreq / old_sfreq))
        slicer = [slice(None)] * x.ndim
        slicer[axis] = slice(new_npad, -new_npad)
        x_resampled = x_resampled[tuple(slicer)]
        
        # Ensure the resampled data has the expected target length.
        if x_resampled.shape[axis] != target_len:
            x_resampled = signal.resample(x_resampled, target_len, axis=axis)
        
        return x_resampled
