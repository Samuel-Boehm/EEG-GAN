#  Authors:	Samuel Böhm <samuel-boehm@web.de>
import os
import  numpy as np
import copy
import joblib

from torch.optim import AdamW
from torch.nn import NLLLoss

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from braindecode.models import deep4

from eeggan.data.preprocess.resample import upsample, downsample
from eeggan.validation.deep4 import train_completetrials
from eeggan.data.create_dataset import load_dataset

# Training the classifier follows the 'braindecode get started' page for trialwise decoding:
# https://braindecode.org/stable/auto_examples/plot_bcic_iv_2a_moabb_trial.html


lr = 1 * 0.01
weight_decay = 0.5 * 0.001

def downsample_for_stage(X, i_stage, max_stage):
    down = downsample(X, 2 ** (max_stage - i_stage), axis=2)
    return upsample(down, 2 ** (max_stage - i_stage), axis=2)


def make_classifier(n_chans, n_classes, n_samples):
    model = deep4(n_chans, n_classes, n_samples, final_conv_length='auto')


def load_deeps4(name: str, stage: int, path: str):
    return joblib.load(os.path.join(path, f'{name}_Deep4_stage{stage}.deep4'))


def train_classifier(train_set, test_set, n_classes, n_chans, deep4_path, n_epochs):

    batch_size = 64

    clf = EEGClassifier(
        model,
        criterion=NLLLoss,
        optimizer=AdamW,
        train_split=predefined_split(valid_set),  # using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
        )

    model = model.cpu().eval()
    return model