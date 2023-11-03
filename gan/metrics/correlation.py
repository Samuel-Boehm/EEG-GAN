# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
from gan.data.batch import batch_data
import pandas as pd

def calculate_correlation_for_condition(batch:batch_data, mapping:dict) -> pd.DataFrame:

    batch.to_numpy()
    conditional_dict = batch.split_batch(mapping)
    
    correlation_dict = {}

    for key in mapping.keys():
        # Calculate the mean over the batch dimension
        real = np.mean(conditional_dict[key]['real'], axis = 0)
        fake = np.mean(conditional_dict[key]['fake'], axis = 0)
        
        correlationCoef = list()
        for i in range(real.shape[0]):
            cc = np.corrcoef(real[i], fake[i])[0, 1]
            correlationCoef.append(cc) 
        correlation_dict[key] = correlationCoef

    return  pd.DataFrame.from_dict(correlation_dict,orient='index').transpose()
