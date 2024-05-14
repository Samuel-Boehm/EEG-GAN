# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from lightning.pytorch.callbacks import ModelCheckpoint


# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    dirpath="my/path/",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
    every_n_epochs=100,
)
