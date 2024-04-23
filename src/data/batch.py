# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import  dataclasses
import numpy as np  
from torch import from_numpy

@dataclasses.dataclass
class batch_data:
    r"""
    Dataclass for storing the data of one batch.
    Makes it easier to pass the data to the metric callbacks.

    Methods: 
    ----------
    
        split_batch: splits the batch according to the mapping into a dictionary
                     with structure: {mapping_key:{'real':real_data, 'fake':fake_data}}

        to_numpy:    moves the data to the cpu and converts it to numpy arrays

        to_tensor:   moves the data to the gpu and converts it to torch tensors
    """
    real: np.ndarray
    fake: np.ndarray
    y_real: np.ndarray
    y_fake: np.ndarray

    def split_batch(self, mapping:dict):
        conditional_dict = {}
        for key in mapping.keys():
            conditional_dict[key] = {
            'real':self.real[self.y_real == mapping[key]],
            'fake':self.fake[self.y_fake == mapping[key]]
            }

        return conditional_dict
    
    def to_numpy(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                continue
            if value.is_cuda:
                value = value.cpu()
            setattr(self, field.name, value.numpy())

    def to_tensor(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            setattr(self, field.name, from_numpy(value))

    def shape(self):
        return self.real.shape