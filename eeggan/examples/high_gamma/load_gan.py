#  Author: Samuel Böhm <samuel-boehm@web.de>

import torch
import joblib
import os
from typing import Tuple

from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.model.builder import ProgressiveModelBuilder


def load_GAN(path: str, stage: int) -> Tuple[Generator, Discriminator, ProgressiveModelBuilder]:
    '''loads a pretrained gan from path and sets a current stage'''
    stateDict = torch.load(os.path.join(path, f'states_stage_{stage}.pt'))
    model_builder =  joblib.load(os.path.join(path, 'model_builder.jblb'))
    
    # Initiate discriminator and generator
    discriminator = model_builder.build_discriminator()
    generator = model_builder.build_generator()
    
    generator.load_state_dict(stateDict['generator'])
    discriminator.load_state_dict(stateDict['discriminator'])
    
    generator.cur_block = stage
    discriminator.cur_block = 6 - stage - 1
    return generator, discriminator, model_builder