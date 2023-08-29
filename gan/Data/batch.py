# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from dataclasses import dataclass
import numpy as np  

@dataclass
class batch_data:
    """
    Dataclass for storing the data of one batch.
    Makes it easier to pass the data to the metric callbacks.
    """
    real: np.ndarray
    fake: np.ndarray
    y_real: np.ndarray
    y_fake: np.ndarray