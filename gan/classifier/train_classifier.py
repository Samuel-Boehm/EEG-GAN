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

dataset_path = os.path.join(data_path, 'generated_stage4')


lr = 1 * 0.01
weight_decay = 0.5 * 0.001
batch_size = 64
n_epochs = 128



_dataset = torch.load(dataset_path)

# Optional: zero pad the dataset: 
pad = torch.nn.ConstantPad1d((50, 50), 0)
X = pad(_dataset[:][0])



train_set = skDataset(X[:4000], _dataset[:4000][1])
valid_set = skDataset(X[4000:], _dataset[4000:][1])


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmarch = True

n_classes = 2
n_chans = 21

input_window_samples = train_set[0][0].shape[1]

model = Deep4Net(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto'
)

if cuda:
    model.cuda()

print(model)

sys.exit()



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

path = f'/home/boehms/eeg-gan/EEG-GAN/temp_plots/deep4_s4.model'

torch.save(model.state_dict(), path)