#  Authors:	Samuel Böhm <samuel-boehm@web.de>
import sys
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
from scipy import linalg
from braindecode.preprocessing import (exponential_moving_standardize,
                                        create_windows_from_events,
                                        preprocess, Preprocessor)
from braindecode.datasets import HGD
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader


def ZCA_whitening(X):
    '''
    Applies zero component analysis whitening to the input X
    X needs to be of shape (trials, channels, datapoints)
    '''
    print(X.shape)
    
    # Zero center data
    xc = X - np.mean(X, axis=0)

    xcov = np.cov(xc, rowvar=True, bias=True)

    # Calculate Eigenvalues and Eigenvectors
    w, v = linalg.eig(xcov) # 
    
    # Create a diagonal matrix
    diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
    diagw = diagw.real.round(4) #convert to real and round off\

    # Whitening transform using ZCA (Zero Component Analysis)

    X_whitened = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
    return X_whitened


class HighGammaModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, sfreq, channels) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sfreq = sfreq
        self.channels = channels
        self.seconds_before_onset = 0.5
        self.seconds_after_onset = 2.5
        self.dataset = None

        self.preprocessor:list[Preprocessor] = [
            Preprocessor('set_eeg_reference', ref_channels='average', projection=False),
            Preprocessor('pick_channels', ch_names=channels),
            Preprocessor(lambda data: np.multiply(data, 1e6)),
            Preprocessor(lambda data: np.clip(data, a_min=-800., a_max=800.)),
            Preprocessor('resample', sfreq=sfreq),
            Preprocessor(exponential_moving_standardize, 
                 factor_new=1e-3, init_block_size=1000),
            Preprocessor(ZCA_whitening, apply_on_array=True, channel_wise=False)
                            ]
        
        super().__init__()
    
    # def prepare_data(self) -> None:
    #    self.raw = HGD(subject_ids=list(range(1, 15)))
        # preprocess(self.raw, self.preprocessor)
        # print('finised preprocessor')
        # self.dataset = create_windows_from_events(
        #    self.dataset, trial_start_offset_samples=0, trial_stop_offset_samples=int(0.5*self.sfreq),
        #    window_size_samples=int(2.5*self.sfreq), window_stride_samples=1, drop_last_window=False, picks=self.channels)
        # self.dataset.save(path=self.data_dir, overwrite=True,)
        

    
    def setup(self, stage: str) -> None:
        self.dataset = load_concat_dataset(
                path=self.data_dir,
                preload=True,
                n_jobs=8
                )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:

        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)
    
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()
    
    def info(self):
        
        if not self.dataset:
            self.dataset = load_concat_dataset(
                path=self.data_dir,
                preload=True,
                n_jobs=8
                )

        df = self.dataset.get_metadata()
        num_labels = len(set(df["target"]))
        n_channels = len(self.channels)
        n_time = int(self.sfreq * (self.seconds_before_onset + self.seconds_after_onset))
        return {'n_channels':n_channels,
                'n_classes':num_labels,
                'n_time':n_time}
    

    

if __name__ == '__main__':

    # safe path for dataset
    dataset_path = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/SchirrmeisterChs'
    
    # targed sfreq
    sfreq = 256

    # Channels to pick
    channels = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    
    High_Gamma__Loader = HighGammaModule(dataset_path, 64, 4, sfreq, channels)
    High_Gamma__Loader.prepare_data()
