# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import  numpy as np
import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset
from torch import from_numpy


class EEGGAN_Dataset(Dataset):
    """
    Dataset class for storing data and labels.
    
        Args:
            tags (list): List of tags for the dataset. Tags are used to split the dataset into different splits.
                         Note that len(tags) must be equal to the number of splits.
            fs (int): Sampling frequency.
        
        Methods:
            add_data: Add data and labels to dataset.
            return_from_tag: Return a DataFrame containing indices for each tag to access data and labels.
            info: Get info about the dataset.
            save: Save dataset to file.
        
        Variables:
            data (np.ndarray): Data.
            target (np.ndarray): Targets.
            splits (pd.DataFrame): DataFrame containing tags for each split. Can be used to split the dataset according to tags.
            fs (int): Sampling frequency.

    """
    def __init__(self, tags:list, fs):
        self.data = np.array([])
        self.target = np.array([])
        self.splits = pd.DataFrame(columns = tags)
        self.splits['idx'] = []
        self.fs = fs
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    def add_data(self, data:np.ndarray, target:np.ndarray, tag_values:list):
        '''
        Add data and labels to dataset.
        Track the start and stop index of the data and labels in the dataset in the 'splits' DataFrame.
        The 'splits' DataFrame can be used to split the dataset according to tags.
        '''
        # Save tags to DataFrame: 
        tag_values.append(list(range(self.data.shape[0], self.data.shape[0] + data.shape[0])))
        self.splits = pd.concat([self.splits, pd.DataFrame([tag_values], columns=self.splits.columns)], ignore_index=True)
        
        # Append Data and Label arrays:
        self.data = np.concatenate((self.data, data), axis=0) if self.data.size else data
        self.target = np.concatenate((self.target, target), axis=0) if self.target.size else target
        
    def return_from_tag(self, tag:str) -> pd.DataFrame:
        '''
        Return a DataFrame containing indices for each tag to access data and labels. 
        '''

        groups = self.splits.groupby(tag)['idx'].apply(list)
        groups.reset_index()
        for index, items  in groups.items():
                groups[index] = list(itertools.chain.from_iterable(items))
        
        return groups.to_frame().reset_index()
    
    def info(self):
        return {'n_time': self.data.shape[-1],
                'fs': self.fs,
                'n_labels': len(np.unique(self.target))
            }
    
    def save(self, path):
        self.data = from_numpy(self.data)
        self.target = from_numpy(self.target)
        torch.save(self, path)