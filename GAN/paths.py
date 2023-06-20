# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

###############################################################################
# This file sets the global paths for the eeg-gan project and needs to be     #
# configured before you can run the project. The paths determine where the    #
# data is stored and where the results are saved.                             # 
###############################################################################

import os
import sys

# This step is needed to set the correct path for the data module
# most likely there is a better way to do this
sys.path.append(os.path.join(__file__, "Data"))

#### Please configure the following paths: ####

# Path to the data directory, when using Data/Preprcess.py this path will be 
# used to save the preprocessed data. This will also be the path to load the
# data when training the model.
data_path = f'/home/samuelboehm/reworkedGAN/Data'

# Path to the directory where the results will be saved. 
# NOTE: the lightning module will use this path to create a subdirectory.
# This subdirectory will contain versions with checkpoints, logs and hyperparameters.
results_path = 'home/samuelboehm/reworkedGAN'
