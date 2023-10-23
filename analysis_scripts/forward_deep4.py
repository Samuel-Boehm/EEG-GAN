# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch
import os
import numpy as np
from braindecode.models.deep4 import Deep4Net
from gan.paths import data_path
from torch import nn
from gan.data.utils import downsample_step

dataset_path = os.path.join(data_path, 'clinical')
dataset = torch.load(dataset_path)

x = downsample_step(dataset[:][0])

# Optional: zero pad the dataset: 
pad = torch.nn.ConstantPad1d((50, 50), 0)
padded = pad(x)

# Load the pretrained model
model_path = "/home/boehms/eeg-gan/EEG-GAN/temp_plots/deep4_s4.model"

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

n_classes = 2
n_chans = 21

print('-----', padded.shape)

input_window_samples = padded.shape[2]

model = Deep4Net(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto'
)
model.load_state_dict(torch.load(model_path))
model.eval()

# Do a forward pass through the model

idx = np.random.choice(len(dataset), size=100, replace=False)

X = padded[idx]
y = dataset[idx][1]

outputs = model(X).argmax(dim=1)

# Calculate the accuracy
accuracy = (outputs == y).float().mean().item()
print(f"Accuracy: {accuracy:.4f}")