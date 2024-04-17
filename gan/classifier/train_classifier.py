# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>
import torch 
import numpy as np
from braindecode.models import Deep4Net
from braindecode.classifier import EEGClassifier

from omegaconf import DictConfig
import hydra

from skorch.callbacks import WandbLogger

from gan.data.datamodule import ProgressiveGrowingDataset

@hydra.main(config_path="configs", config_name="classifier_config")
def main(cfg: DictConfig):
    raise NotImplementedError("This script is not implemented yet.")