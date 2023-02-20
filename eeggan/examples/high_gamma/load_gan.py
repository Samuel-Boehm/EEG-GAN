#  Author: Samuel Böhm <samuel-boehm@web.de>

import torch
import joblib
import os
from typing import Tuple
from collections import OrderedDict


from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.model.builder import ProgressiveModelBuilder


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
    return generator, discriminator, model_builder


def load_spectral_GAN(path: str, stage: int) -> Tuple[Generator, Discriminator, Discriminator, ProgressiveModelBuilder]:
    '''loads a pretrained gan from path and sets a current stage'''
    stateDict = torch.load(os.path.join(path, f'states_stage_{stage}.pt'))
    model_builder =  joblib.load(os.path.join(path, 'model_builder.jblb'))

    # original saved file with DataParallel
    generator_state_dict = OrderedDict()
    for k, v in stateDict['generator'].items():
        name = k.replace('module.', '') 
        generator_state_dict[name] = v
    
    discriminator_state_dict = OrderedDict()
    for k, v in stateDict['discriminator'].items():
        name = k.replace('module.', '') 
        discriminator_state_dict[name] = v

    spectral_discriminator_state_dict = OrderedDict()
    for k, v in stateDict['spectral_discriminator'].items():
        name = k.replace('module.', '') 
        spectral_discriminator_state_dict[name] = v


    # Initiate discriminator and generator
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()
    spectralDiscriminator = model_builder.build_spectral_discriminator()
    
    generator.cur_block = stage
    discriminator.cur_block = 6 - stage - 1
    spectralDiscriminator.cur_block = 6 - stage - 1
    
    generator.load_state_dict(generator_state_dict)
    discriminator.load_state_dict(discriminator_state_dict)
    spectralDiscriminator.load_state_dict(spectral_discriminator_state_dict)


    return generator, discriminator, spectralDiscriminator, model_builder