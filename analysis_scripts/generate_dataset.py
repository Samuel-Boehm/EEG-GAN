# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.utils import generate_data
from gan.paths import data_path
import os

if __name__ == "__main__":
    
    path = "EEGGAN/twlqedvh"
    stage = 4
    ds = generate_data(path, stage, 10000)
    ds.save(os.path.join(data_path, 'generated_stage4'))