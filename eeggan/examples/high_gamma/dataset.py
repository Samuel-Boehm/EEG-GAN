#  Author: Kay Hartmann <kg.hartma@gmail.com>
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from eeggan.data.dataset import SignalAndTarget, Data, Dataset


@dataclass
class HighGammaDataset(Dataset[np.ndarray]):
    """
    High Gamma Dataset container

    Args:
        train_data (SignalAndTarget): train dataset
        test_data (SignalAndTarget): test dataset
        n_time (int): number of time points
        channels (List[str]): used EEG channel names
        classes (List[str]): used labels
        fs (float): sampling rate
    """

    def __init__(self, train_data: Data[np.ndarray], test_data: Data[np.ndarray], n_time: int, channels: List[str],
                 classes: List[str], fs: float):
        super().__init__(train_data, test_data)

        self.n_time = n_time
        self.channels = channels
        self.classes = classes
        self.fs = fs

    def __str__(self):
        description = f'--HighGammaDataset--\n' \
                      f'\n' \
                      f'Train Data:\n' \
                      f'X shape: {self.train_data.X.shape} y shape {self.train_data.y.shape}\n' \
                      f'\n' \
                      f'Test Data:\n' \
                      f'X shape: {self.test_data.X.shape} y shape {self.test_data.y.shape}\n' \
                      f'\n' \
                      f'n_time: {self.n_time}\n' \
                      f'channels: {self.channels}\n' \
                      f'classes: {self.classes}\n' \
                      f'fs: {self.fs}\n'
        return description
