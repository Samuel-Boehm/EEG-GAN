# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import sys
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from braindecode.models.deep4 import Deep4Net
from braindecode import EEGClassifier
from gan.paths import data_path
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from skorch.dataset import Dataset as skDataset
import torch
import sys
import os

dataset_path = os.path.join(data_path, 'clinical')
dataset = torch.load(dataset_path)


lr = 1 * 0.01
weight_decay = 0.5 * 0.001
batch_size = 64
n_epochs = 128


n_classes = 2
n_chans = 21

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmarch = True


if __name__ == '__main__':

    for i in range(1, 15):
        
        train_idx, val_idx = dataset.splits['idx'][dataset.splits.subject == i].values
        
        train_set = skDataset(dataset[train_idx][0], dataset[train_idx][1])
        valid_set = skDataset(dataset[val_idx][0], dataset[val_idx][1])
        
        input_window_samples = train_set[0][0].shape[1]

        model = Deep4Net(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length='auto'
        )

        if cuda:
            model.cuda()



        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(valid_set),  # using valid_set for validation
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            ],
            device=device,
        )


        clf.fit(train_set, y=None, epochs=n_epochs)

        model.cpu()

        path = f'/home/boehms/eeg-gan/EEG-GAN/temp_plots/Deep4/sub_{i}.model'

        torch.save(model.state_dict(), path)
