# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
import os 
import torch
from gan.paths import results_path, data_path
from gan.data.DataSet import EEGGAN_Dataset


path = "EEGGAN/4vj37rcg"
stage = 5



model = GAN.load_from_checkpoint(
                checkpoint_path=os.path.join(results_path, path, "checkpoints/last.ckpt"),
                map_location=torch.device('cpu'),
                )


model.eval()
model.freeze()

# Set Stage:
model.generator.set_stage(stage)
model.critic.set_stage(stage)

model.generator.fading = False
model.critic.fading = False

fs_stage = int(model.hparams.fs/2**(model.hparams.n_stages-(model.current_stage-1)))


# generate noise and labels:
n_samples = 5000
batch_size = 128

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
ds.save(os.path.join(data_path, 'generated'))

