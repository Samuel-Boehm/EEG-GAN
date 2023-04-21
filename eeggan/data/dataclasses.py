#  Author: Kay Hartmann <kg.hartma@gmail.com>

from typing import Iterable, Tuple, TypeVar, Generic
from dataclasses import dataclass
import numpy as np

from eeggan.data.preprocess.normalize import normalize_data

T = TypeVar('T')


@dataclass
class Data(Iterable, Generic[T]):

    X: T 
    y: T

    def __post_init__(self):
        self.index_dict: dict = {}

    def __iter__(self) -> Tuple[T, T, T]:
        for i in range(len(self.X)):
            yield self[i]

    def __getitem__(self, index: int) -> Tuple[T, T, T]:
        return self.X[index], self.y[index],

    def __len__(self) -> int:
        return len(self.X)
    
    def return_subset(self, n_samples:int):
        '''
        Return a subsample of the Dataset after a list of indices 
        '''
        assert n_samples % 2 == 0, "To have balanced subsets n_samples needs to be an even number"
        y0 = np.where(self.y==0)[0]
        y1 = np.where(self.y==1)[0]

        r1 = np.random.choice(y0, n_samples//2, replace=False)
        r2 = np.random.choice(y1, n_samples//2, replace=False)

        index = np.concatenate((r1, r2), axis=0)
        
        return Data(self.X[index], self.y[index])
    
    def return_subject(self, subject):
        '''Return a subject after a list of subject IDs'''
        first = self.index_dict[subject][0]
        last = self.index_dict[subject][1]
        return Data(self.X[first:last], self.y[first:last])

    def add_data(self, X, y, subject):
        '''Add data to an existing datavariable'''
        if self.X.shape[0] == 0:
            self.X = X
            self.y = y
            self.index_dict[subject] = (0, self.X.shape[0])
        else:
            assert X.shape[1:] == self.X.shape[1:]
            self.index_dict[subject] = (int(self.X.shape[0]), int(self.X.shape[0] + X.shape[0]))
            self.X = np.concatenate((self.X, X), axis=0)
            self.y = np.concatenate((self.y, y), axis=0)

    def prepare_data(self, normalize = False):
        '''Normalize data and map labels'''
        
        if normalize:
            self.X = normalize_data(self.X)

        # Here the labels are mapped so that we have labels in range 0 to n_unique_labels
        unique_labels = list(set(self.y))
        self.y = np.array(list(map(lambda x: unique_labels.index(x), self.y)))

            
@dataclass
class Dataset(Generic[T]):
    train_data: Data[T]
    test_data: Data[T]

    def add_metadata(self, n_time, channels, classes, fs):

        self.n_time = n_time
        self.channels = channels
        self.classes = classes
        self.fs = fs
