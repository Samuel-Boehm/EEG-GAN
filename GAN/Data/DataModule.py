#  Authors:	Samuel Böhm <samuel-boehm@web.de>
import sys
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from lightning import LightningDataModule
# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HighGammaModule(LightningDataModule):
    '''
    DataModule for HighGamma data. Loads data from a pickle file. 
    Pickle file needs to contain a class with at least following attributes:
        data: Array of shape (n_samples, n_channels, n_timesteps)
        __len__: Returns number of samples
        __getitem__: Returns sample at index i
    
    Note: This DataModule was specifically designed for the EEG-GAN project.

    Args:
        data_dir (str): Path to pickle file
        n_stages (int): Number of stages for progressive growing
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader 
    
    Methods:
        set_stage: Resamples data for given stage according the number of stages (n_stages)
    '''

    def __init__(self, data_dir:str, n_stages:int, batch_size:int, num_workers:int) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_stages = n_stages
        super().__init__()

    def setup(self, stage=None) -> None:
        self.ds = torch.load(self.data_dir)
        self.orignal_set = torch.clone(self.ds.data)

    def train_dataloader(self):
        return DataLoader(self.ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)
     
    def test_dataloader(self):
        return super().test_dataloader()
    
    def set_stage(self, stage: int):

        # Resample original set for stage
        x = torch.unsqueeze(self.orignal_set, 0)
        for i in range(int(self.n_stages - stage)):
            x = nn.functional.interpolate(x, scale_factor=(1, 0.5), mode="bicubic")
        x = torch.squeeze(x, 0)
        self.ds.data = x
