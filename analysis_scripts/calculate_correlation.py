# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import numpy as np
import argparse
import pandas as pd
import os 
from tqdm import tqdm
import joblib

from gan.data.dataset import EEGGAN_Dataset
from gan.data.utils import downsample_step

CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'M1', 'M2']

MAPPING = {'right': 0, 'rest': 1}

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--gan_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Results/Thesis/')
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--stage', type=int, required=True)
parser.add_argument('--subject', type=int, required=False)
parser.add_argument('--data_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Data/Thesis/')
parser.add_argument('--deep4_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Models/Thesis/')
parser.add_argument('--downsample', type=bool, required=False, default=False)
args = parser.parse_args()

gan_path = os.path.join(args.gan_path, args.name)



def calculate_Corelation(real:EEGGAN_Dataset, fake:EEGGAN_Dataset, mapping=MAPPING):


    conditional_dict = {}

    for key in mapping.keys():
        conditional_dict[key] = [
            real.data[real.target == mapping[key]],
            fake.data[fake.target == mapping[key]]
            ]
    correlation_dict = {}
    for key in mapping.keys():
        real = np.mean(conditional_dict[key][0], axis = 0)
        fake = np.mean(conditional_dict[key][1], axis = 0)
        
        correlationCoef = list()
        for i in tqdm(range(21)):
            cc = np.corrcoef(real[i], fake[i])[0, 1]
            correlationCoef.append(cc) 
        correlation_dict[key] = correlationCoef
    return  pd.DataFrame.from_dict(correlation_dict,orient='index').transpose()
                





if __name__ == '__main__':
    print('Running', args.name)
    
    if args.name == 'torch_cGAN': 
        real = joblib.load(os.path.join(args.data_path, f'tGAN_real_stage{args.stage}.dataset'))
        fake = joblib.load(os.path.join(args.data_path, f'tGAN_fake_stage{args.stage}.dataset'))

    else:
        real, fake = create_balanced_datasets(gan_path, args.data_path, args.data_name, args.stage, None, False, n_samples=3000)

    if args.downsample == True:
        real.X = downsample(real.X, 2, axis=2)
        fake.X = downsample(fake.X, 2, axis=2)



    df = calculate_Corelation(real, fake)
    df.set_index = CHANNELS
    means = list()
    stds = list()

    for i in MAPPING.keys():
        means.append(df[i].mean())
        stds.append(df[i].std())

    df.loc['mean'] = means
    df.loc['std'] = stds

    df.to_csv(f'/home/boehms/eeg-gan/EEG-GAN/EEG-GAN/analysis_scripts/outputs/{args.name}_correlation_stage{args.stage}.csv')