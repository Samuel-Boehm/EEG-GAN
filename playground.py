import torch 

import mne
mne.set_log_level('ERROR')

d_ = torch.load('/home/samuelboehm/EEG-GAN/datasets/clinical/S1.pt')

from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import preprocess, Preprocessor
import numpy as np



def change_type(X: np.ndarray, out_type: str) -> np.ndarray:
    # MNE expects the data to be of type float64. This helper function changes the type of the input data to float64.
    if out_type == 'float64':
        return X.astype('float64')
    elif out_type == 'float32':
        return X.astype('float32')
    else:
        raise ValueError(f"Unknown type {out_type}")

base_sfreq = 256
sfreq = int(base_sfreq // 2**5)

print(sfreq)

preprocessors = [Preprocessor(change_type, out_type='float64')]
data = preprocess(d_, preprocessors, n_jobs=1)

preprocessors = [Preprocessor('resample', sfreq=sfreq, npad=0)]
data = preprocess(data, preprocessors, n_jobs=1)

print(data.datasets[0][0][0].shape)


def _resamp_ratio_len(up, down, n):
    ratio = float(up) / down
    return ratio, max(int(round(ratio * n)), 1)