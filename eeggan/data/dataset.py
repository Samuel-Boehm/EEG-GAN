#  Author: Samuel Böhm <samuel-boehm@web.de>

from typing import Iterable, Tuple, TypeVar, Generic
from dataclasses import dataclass
import numpy as np

T = TypeVar('T')


@dataclass
class SignalAndTarget(object):
    """
    Data container class inspired by the eponymous braindecode class.

    The input signal is saved together with an array of labels. 
    If a subject if given a dictionary is created, saving the start and stop index of
    the subjects data block. This can be  used to return individual subjects later on. 

    Source: 
    https://github.com/robintibor/braindecode/blob/master/braindecode/datautil/signal_target.py#L1-L16
    
    -----------------------------------------------------------------
    
    X (np.ndarray) : 3D array
        The input signal per trial.
    y (np.ndarray) : 1D array
        Labels for each trial.
    subject (int): subject ID
    """
    
    def __init__(self, X:np.ndarray, y:np.ndarray, subject:int):
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.subject: int = subject
        self.index_dict: dict = {}
        if subject:
            self.index_dict[subject] = (0, self.X.shape[0])

        assert len(X) == len(y)

    def __str__(self):
        return f'shape X:{self.X.shape} shape y:{self.y.shape}'

    def add_data(self, X, y, subject):
        
        """
        Adds another X, y block to the data. 
        """
        
        if self.X.shape[0] == 0:
            self.X = X
            self.y = y
            self.index_dict[subject] = (0, self.X.shape[0])

        else:
            assert X.shape[1:] == self.X.shape[1:]
            self.index_dict[subject] = (int(self.X.shape[0]), int(self.X.shape[0] + X.shape[0]))
            self.X = np.concatenate((self.X, X), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)
            


@dataclass
class Data(SignalAndTarget, Iterable, Generic[T]):
    """
    Base datastructure for the EEG-GAN project.
    This structure holds the data block X and the labels
    in integer and onehot encoding. 
    """
    
    X: T 
    y: T
    y_onehot: T 
    

    def __post_init__(self):
        self.index_dict: dict = {}


    def __iter__(self) -> Tuple[T, T, T]:
        for i in range(len(self.X)):
            yield self[i]

    def __getitem__(self, index: int) -> Tuple[T, T, T]:
        return self.X[index], self.y[index], self.y_onehot[index],

    def __len__(self) -> int:
        return len(self.X)
    
    def subset(self, n_samples:int):
        """
        Return a subset of the Dataset after a list of indices (used to draw random samples) 
        """
        assert n_samples % 2 == 0, "To have balanced subsets n_samples needs to be an even number"
        y0 = np.where(self.y==0)[0]
        y1 = np.where(self.y==1)[0]

        r1 = np.random.choice(y0, n_samples//2, replace=False)
        r2 = np.random.choice(y1, n_samples//2, replace=False)

        index = np.concatenate((r1, r2), axis=0)
        
        return Data(self.X[index], self.y[index], self.y_onehot[index])
    
    def return_subject(self, subject):
        """
        Return subject(s) after a list of subject IDs
        """
        first = self.index_dict[subject][0]
        last = self.index_dict[subject][1]
        return Data(self.X[first:last], self.y[first:last], self.y_onehot[first:last])

    def add_data(self, X, y, y_onehot, subject):
        """
        Add another block of data
        """
        if self.X.shape[0] == 0:
            self.X = X
            self.y = y
            self.y_onehot = y_onehot
            self.index_dict[subject] = (0, self.X.shape[0])
        else:
            assert X.shape[1:] == self.X.shape[1:]
            self.index_dict[subject] = (int(self.X.shape[0]), int(self.X.shape[0] + X.shape[0]))
            self.X = np.concatenate((self.X, X), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)
            self.y_onehot = np.concatenate((self.y_onehot, y_onehot), axis=0)
            
@dataclass
class Dataset(Generic[T]):
    """
    This dataclass hols a train and test data Data object
    """
    train_data: Data[T]
    test_data: Data[T]
