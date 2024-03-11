# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import  numpy as np
import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset
from torch import from_numpy


class EegGanDataset(Dataset):
    """
    Dataset class for storing data and labels.
    
        Args:
            tags (list): List of tags for the dataset. Tags are used to split the dataset into different splits.
                         Note that len(tags) must be equal to the number of splits.
            interval_times: Tuple with start and stop seconds relative to the trigger in seconds.
                            For example (-.5, 2.) will cout out 0.5 seconds before the trigger until 2 seconds after
                            the trigger. 
            fs (int): Sampling frequency.
            mapping (dict): label mapping from strings to integers. Needed to later trace what lable ment what. 
            channels (list): Channel names, we safe them here to make plotting more easy later on
        
        Methods:
            add_data: Add data and labels to dataset.
            select_from_tag: Select a subset of the dataset according to a tag and a selection from the tag.
            tags: Show possible tags and selections.
            info: Get info about the dataset.
            save: Save dataset to file.
        
        Variables:
            data (np.ndarray): Data.
            target (np.ndarray): Targets.
            splits (pd.DataFrame): DataFrame containing tags for each split. Can be used to split the dataset according to tags.
            fs (int): Sampling frequency.

    """
    def __init__(self, tags:list, interval_times:tuple, fs:float, mapping:dict, channels:list):
        self.data = np.array([])
        self.target = np.array([])
        self.splits = pd.DataFrame(columns = tags)
        self.splits['idx'] = []
        self.interval_times = interval_times
        self.fs = fs
        self.mapping = mapping
        self.channels = channels
    
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
        
    def select_from_tag(self, tag:str, selection) -> pd.DataFrame:
        '''
        Return a subset of the dataset according to tag and a selection from the tag.
        ATTENTION: This method changes the dataset in place.

        Arguments:
        ----------
        tag (str): Tag to select subset from.
        selection (str): Selection from tag. To show possible tags and selections use the tags() method.

        Example:
        ----------
        Return a specific subject from the dataset:
        >>> ds = torch.load('path/to/dataset')
        >>> ds.tags()
        out:
            Possible tags:
            ------------------
            subject : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            session : ['session_0']
            split : ['test', 'train']
        >>> ds.return_from_tag('subject', 4)

        '''
        train_idx, test_idx = self.splits['idx'][self.splits[tag] == selection].values
        idx = np.concatenate((train_idx, test_idx))
        self.data = self.data[idx]
        self.target = self.target[idx]

        return self.data, self.target

    def tags(self):
        print('Possible tags:'
              '\n------------------')
        for tag in self.splits.columns:
            if tag != 'idx':
                print(tag, ':', list(np.unique(self.splits[tag])))



    def info(self):
        return {'n_time': self.data.shape[-1],
                'fs': self.fs,
                'n_labels': len(np.unique(self.target))
            }
    
    def save(self, path):
        self.data = from_numpy(self.data)
        self.target = from_numpy(self.target)
        torch.save(self, path)