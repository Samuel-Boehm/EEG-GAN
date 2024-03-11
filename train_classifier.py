from braindecode.models.deep4 import Deep4Net
from braindecode import EEGClassifier
from gan.paths import data_path
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from skorch.dataset import Dataset
import torch
import os

from constants import BASEDIR

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

if __name__ == '__main__':

    DATA_PATH = os.path.join(BASEDIR, 'clinical')

    LR = 1 * 0.01
    WEIGHT_DECAY = 0.5 * 0.001
    BATCH_SIZE = 64
    N_EPOCHS = 128

    data = torch.load(DATA_PATH)

    # Optional: zero pad the dataset: 
    # We might need this if we work with low resoultuons
    # pad = torch.nn.ConstantPad1d((50, 50), 0)
    #X = pad(_dataset[:][0])