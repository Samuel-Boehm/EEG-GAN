#  Author: Samuel Boehm <samuel-boehm@web.de>

import sys
# setting path
sys.path.append('/home/samuelboehm/boehms/eeg-gan/EEG-GAN/EEG-GAN')

import numpy as np
import torch 
from torch import Tensor
import os
import joblib
from collections import OrderedDict


from eeggan.examples.high_gamma.make_data import load_dataset
from eeggan.cuda import to_cuda, init_cuda
from eeggan.examples.high_gamma.dataset import HighGammaDataset
from eeggan.data.preprocess.resample import downsample
from eeggan.data.dataset import Data
from numpy.random.mtrand import RandomState


def balance_dataset(data: HighGammaDataset):
    label_counts =  np.unique(data.train_data.y, return_counts=True)
    smallest_dataset = label_counts[1].min()
    idx = list()
    for i in label_counts[0]:
        idx.append(np.where(data.train_data.y == i)[0][:smallest_dataset])

    idx = np.array(idx).flatten()
    data.train_data.y = data.train_data.y[idx]
    data.train_data.y_onehot = data.train_data.y_onehot[idx]
    data.train_data.X = data.train_data.X[idx]
    return data 

def load_GAN(path: str, stage: int):
    '''loads a pretrained gan from path and sets a current stage'''
    stateDict = torch.load(os.path.join(path, f'states_stage_{stage}.pt'))
    
    # original saved file with DataParallel
    generator_state_dict = OrderedDict()
    for k, v in stateDict['generator'].items():
        name = k.replace('module.', '') # removing ‘.moldule’ from key
        generator_state_dict[name] = v
    
    discriminator_state_dict = OrderedDict()
    for k, v in stateDict['discriminator'].items():
        name = k.replace('module.', '') # removing ‘.moldule’ from key
        discriminator_state_dict[name] = v

    model_builder =  joblib.load(os.path.join(path, 'model_builder.jblb'))
    
    # Initiate discriminator and generator
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()
    
    generator.load_state_dict(generator_state_dict)
    discriminator.load_state_dict(discriminator_state_dict)
    
    generator.cur_block = stage
    discriminator.cur_block = 6 - stage - 1
    return generator, discriminator


def create_fake_data(model_path, stage, n_samples, balanced = True):
    generator, discriminator = load_GAN(model_path, stage - 1)

    latent, y_fake, y_onehot_fake = generator.create_latent_input(RandomState(), n_samples, balanced)

    X_fake =  generator(latent, y=y_fake).detach().numpy()
    return Data(X_fake, y_fake, y_onehot_fake)

def create_fake_data_old(model_path, stage, n_samples, balanced = True):
    generator, discriminator = load_GAN(model_path, stage - 1)

    latent, y_fake, y_onehot_fake = generator.create_latent_input(RandomState(), n_samples, balanced)

    X_fake =  generator(latent, y=y_fake, y_onehot=y_onehot_fake).detach().numpy()
    return Data(X_fake, y_fake, y_onehot_fake)


def create_balanced_datasets(gan_path, data_path, data_name, stage, subject=None, return_on_cuda=False, n_samples=256):
    # loading real data...
    real = load_dataset(data_name, data_path)
    
    if subject:
        real.train_data = real.train_data.return_subject(subject)
        real.test_data = real.test_data.return_subject(subject)

    real.train_data.X = np.concatenate((real.train_data.X[:], real.test_data.X[:]), axis=0)
    real.train_data.y = np.concatenate((real.train_data.y[:], real.test_data.y[:]), axis=0)
    real.train_data.y_onehot = np.concatenate((real.train_data.y_onehot[:], real.test_data.y_onehot[:]), axis=0)

    # If there is a disbalance in the real dataset drop rows to balance the real dataset
    real = balance_dataset(real)

    real = real.train_data.subset(n_samples)
    
    real.X = downsample(real.X, factor=2 ** (6-stage), axis =2)

    # generating fake data...

    n_samples = real.X.shape[0]

    # Data is generated in batches of 1000 to make it work on lesser memory
    batches = []
    while n_samples > 1000:
        batches.append(1000)
        n_samples -= 1000
    batches.append(n_samples)

    fake = create_fake_data(gan_path, stage, batches[0])

    for i in batches[1:]:
        temp_fake = create_fake_data(gan_path, stage, i)
        fake.add_data(temp_fake.X, temp_fake.y, temp_fake.y_onehot, 1)

    if return_on_cuda:
        init_cuda()
        real = Data(*to_cuda(Tensor(real.X), Tensor(real.y), Tensor(real.y_onehot)))
        fake = Data(*to_cuda(Tensor(fake.X), Tensor(fake.y), Tensor(fake.y_onehot)))
    
    return real, fake