# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
import os 
import torch
from gan.paths import results_path, data_path
from gan.data.dataset import EEGGAN_Dataset
from gan.data.batch import batch_data
from gan.data.utils import downsample_step
import numpy as np



path = "EEGGAN/twlqedvh"
stage = 4
n_samples = 520


def generate_data(model_path, stage, n_samples, batch_size=128,):
    # Load Model:
    model = GAN.load_from_checkpoint(
                checkpoint_path=os.path.join(results_path, model_path, "checkpoints/last.ckpt"),
                map_location=torch.device('cpu'),
                )
    model.eval()
    model.freeze()

    # Set Stage:
    model.generator.set_stage(stage)
    model.critic.set_stage(stage)

    # Disable fading:
    model.generator.fading = False
    model.critic.fading = False

    # Calculate fs in current stage:
    fs_stage = int(model.hparams.fs/2**(model.hparams.n_stages-(model.cur_stage-1)))
    
    # generate data in batches to avoid memory issues:
    for batch in range(n_samples // batch_size):
        z = torch.randn(batch_size, model.hparams.latent_dim)
        y_fake_batch = torch.randint(low=0, high=model.hparams.n_classes,
                                size=(batch_size,), dtype=torch.int32)

        # generate fake batch:
        X_fake_batch = model.forward(z, y_fake_batch)

        if batch == 0:
            X_fake = X_fake_batch
            y_fake = y_fake_batch
        else:
            X_fake = torch.cat((X_fake, X_fake_batch), dim=0)
            y_fake = torch.cat((y_fake, y_fake_batch), dim=0)

    if n_samples % batch_size != 0:
        z = torch.randn(n_samples % batch_size, model.hparams.latent_dim)
        y_fake_batch = torch.randint(low=0, high=model.hparams.n_classes,
                                size=(n_samples % batch_size,), dtype=torch.int32)

        # generate fake batch:
        X_fake_batch = model.forward(z, y_fake_batch)

        X_fake = torch.cat((X_fake, X_fake_batch), dim=0)
        y_fake = torch.cat((y_fake, y_fake_batch), dim=0)


    X_fake = X_fake.detach().numpy()
    y_fake = y_fake.detach().numpy()

    ds = EEGGAN_Dataset(['session',], fs_stage)

    ds.add_data(X_fake, y_fake, [1,])
    return ds

real = torch.load(os.path.join(data_path, 'clinical'))

for i in range(1):
    real.data = downsample_step(real.data)

train_idx, test_idx = real.splits['idx'][real.splits.subject == 3].values

# For GAN training we pool all data together:
idx = np.concatenate((train_idx, test_idx))

real.data = real.data[idx]
real.target = real.target[idx]


fake = generate_data(path, stage, n_samples)

print(fake.data.shape, real.data.shape)

batch = batch_data(real.data, fake.data,
                   real.target, fake.target)