from Model.GAN import GAN
from pytorch_lightning.trainer import Trainer
import torch
from braindecode.datautil import load_concat_dataset
from Data.DataModule import HighGammaModule as HDG
from Handler.ProgressionHandler import Scheduler

dataset_path = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/clinical2'

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']


dm = HDG(dataset_path, batch_size=32, num_workers=8, sfreq=256, channels=channels)

# Get Data relevant parameters from the data module
GAN_PARAMS = {
    'n_channels':len(channels),
    'n_classes':4,
    'n_time':dm.n_time,
    'n_stages':6,
    'n_filters':120,
    'latent_dim':210,
    'epochs_per_stage':1
    }

TRAINER_PARAMS = {
    'max_epochs': GAN_PARAMS['epochs_per_stage'] * GAN_PARAMS['n_stages'], 
    
    }

def main():
    model = GAN(**GAN_PARAMS)

    trainer = Trainer(
            max_epochs=GAN_PARAMS['epochs_per_stage'] * GAN_PARAMS['n_stages'],
            reload_dataloaders_every_n_epochs=GAN_PARAMS['epochs_per_stage'],
            callbacks=[Scheduler()],
    )
    
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()