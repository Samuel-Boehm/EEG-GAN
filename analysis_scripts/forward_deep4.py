import torch
import os
import numpy as np
from braindecode.models.deep4 import Deep4Net
from gan.paths import data_path

dataset_path = os.path.join(data_path, 'generated')
dataset = torch.load(dataset_path)

# Load the pretrained model
model_path = "/home/boehms/eeg-gan/EEG-GAN/temp_plots/deep4.model"

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

n_classes = 2
n_chans = 21

input_window_samples = dataset[0][0].shape[1]

model = Deep4Net(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto'
)
model.load_state_dict(torch.load(model_path))
model.eval()

# Do a forward pass through the model

X, y = dataset[np.random.choice(len(dataset), size=100, replace=False)]

outputs = model(X).argmax(dim=1)

# Calculate the accuracy
accuracy = (outputs == y).float().mean().item()
print(f"Accuracy: {accuracy:.4f}")