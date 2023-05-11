from braindecode.datautil import load_concat_dataset
from braindecode.models.deep4 import Deep4Net
from braindecode import EEGClassifier

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

import torch

lr = 1 * 0.01
weight_decay = 0.5 * 0.001
batch_size = 64
n_epochs = 128

dataset_path = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/SchirrmeisterChs'

CHANNELS = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']

windows_dataset = load_concat_dataset(
    path=dataset_path,
    preload=True,
    target_name=None,
)


splitted = windows_dataset.split('run')
train_set = splitted['train']
valid_set = splitted['test']


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmarch = True

n_classes = 4
n_chans = len(CHANNELS)
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