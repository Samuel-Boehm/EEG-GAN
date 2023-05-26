from GAN import GAN
from pytorch_lightning.trainer import Trainer
import torch
from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader
from DataModule import HighGammaModule

dataset_path = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/SchirrmeisterChs'

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

dm = HighGammaModule(dataset_path, batch_size=32, num_workers=8, sfreq=256,  channels=channels)


    
# Get Data relevant parameters from the data module
GAN_PARAMS = dm.info()

# Add further Network Parameters
GAN_PARAMS.update({
    'n_stages':6,
    'n_filters':120,
    'latent_dim':210,
    'epochs_per_stage':500
    })


def main():
    
    model = GAN(**GAN_PARAMS)
    # print("####"*10, "generator", "####"*10)
    # print(model.generator)
    # print("####"*10, "critic", "####"*10)
    # print(model.critic)

    print('Generator Stage', model.generator.cur_stage)
    print('Critic Stage', model.critic.cur_stage)

    trainer = Trainer(max_epochs=model.progress.epochs_total, reload_dataloaders_every_n_epochs=GAN_PARAMS['epochs_per_stage'])
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()