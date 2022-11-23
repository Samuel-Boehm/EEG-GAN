#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import Iterable, Tuple, TypeVar, Generic
from dataclasses import dataclass
import numpy as np

T = TypeVar('T')


@dataclass
class SignalAndTarget(object):
    """
    Simple data container class.
    Parameters
    ----------
    X (np.ndarray) : 3D array
        The input signal per trial.
    y (np.ndarray) : 1D array
        Labels for each trial.
    """
    def __init__(self, X, y):
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        assert len(X) == len(y)

    def __str__(self):
        return f'shape X:{self.X.shape} shape y:{self.y.shape}'

    def add_data(self, X, y):
        if self.X.shape[0] == 0:
            self.X = X
            self.y = y
        else:
            assert X.shape[1:] == self.X.shape[1:]
            self.X = np.concatenate((self.X, X), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)


@dataclass
class Data(SignalAndTarget, Iterable, Generic[T]):

    X: T 
    y: T
    y_onehot: T

    def __iter__(self) -> Tuple[T, T, T]:
        for i in range(len(self.X)):
            yield self[i]

    def __getitem__(self, index: int) -> Tuple[T, T, T]:
        return self.X[index], self.y[index], self.y_onehot[index],

    def __len__(self) -> int:
        return len(self.X)
    
    def subset(self, index):
        '''
        Return a subsample of the Dataset after a list of indices 
        '''
        return Data(self.X[index], self.y[index], self.y_onehot[index])


@dataclass
class Dataset(Generic[T]):
    train_data: Data[T]
    test_data: Data[T]
