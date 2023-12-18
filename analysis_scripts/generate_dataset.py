# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.utils import generate_data
from gan.paths import data_path
import os

if __name__ == "__main__":
    
    path = "x6hrdnur"
    stage = 5
    ds = generate_data(path, stage, 1000)
    ds.save(os.path.join(data_path, 'subject1'))