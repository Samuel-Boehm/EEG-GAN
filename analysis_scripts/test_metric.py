# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

# This file is used to test  metrics.

from braindecode.models.deep4 import Deep4Net
from legacy_code.metrics.IntermediateOutput import IntermediateOutputWrapper
from legacy_code.data.batch import batch_data

from legacy_code.metrics.correlation import calculate_correlation_for_condition
from legacy_code.visualization.time_domain_plot import plot_time_domain
from legacy_code.paths import data_path

import torch

import os

mapping = {'right': 0, 'rest': 1}

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

real = torch.load(os.path.join(data_path, 'clinical'))

real.tags()

