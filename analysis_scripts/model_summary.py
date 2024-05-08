# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from pandas.core.frame import DataFrame
import wandb
from legacy_code.model.gan import GAN
import numpy as np

path = "EEGGAN/hgong7rh" #eeggan/
api = wandb.Api()

run = api.run(path)
config = run.config

model = GAN(**config)

model.generator.set_stage(3)
model.critic.set_stage(3)


def count_parameters(model):
    df = DataFrame(columns=['name', 'params'])
    total_params = 0
    for i, params in enumerate(model.named_parameters()):
        name, parameter = params
        if not parameter.requires_grad: continue
        params = parameter.numel()
        df.loc[i] = [name, params]
        total_params+=params
    print(df)
    return total_params

count_parameters(model.generator)


n_time = 768
n_stages = 3

n_time_first_layer = int(np.floor(n_time / 2 ** (n_stages-1)))

print(n_time_first_layer)

latent_dim = 210
embedding_dim = 10
n_filters = 120


print(latent_dim + embedding_dim)

print(n_filters * n_time_first_layer)

print((latent_dim + embedding_dim) * (n_filters * n_time_first_layer))

