from typing import Tuple
from src.utils.utils import to_numpy
import hydra
from src.models.gan import GAN
from src.data.datamodule import ProgressiveGrowingDataset
import torch
from omegaconf import DictConfig
from numpy import ndarray

def return_real_and_fake(model_path:str, cfg:DictConfig, stage:int) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    
    checkpoint = list(model_path.glob('*.ckpt'))[0]
    
    n_samples = int(cfg.data.sfreq * cfg.data.length_in_seconds)

    models = dict()

    generator = hydra.utils.instantiate(cfg.model.generator, n_samples=n_samples)
    models['generator'] = generator

    critic = hydra.utils.instantiate(cfg.model.critic, n_samples=n_samples)
    models['critic'] = critic

    if hasattr(cfg.model, 'spectral_critic'):
        spectral_critic = hydra.utils.instantiate(cfg.model.spectral_critic, n_samples=n_samples)
        models['spectral_critic'] = spectral_critic
    
    # Remove target attribute from cfg.model.gan
    if hasattr(cfg.model.gan, '_target_'):
        del cfg.model.gan._target_

    model = GAN.load_from_checkpoint(checkpoint, **models, **cfg.model.gan, optimizer=cfg.model.optimizer, strict=False)
    model.eval()
    model.generator.set_stage(stage)

    ds = ProgressiveGrowingDataset(cfg.data.dataset_name, batch_size=512, n_stages=cfg.trainer.scheduler.n_stages)
    ds.set_stage(stage)

    dl = ds.train_dataloader()

    X_real, y_real = next(iter(dl)) # This returns a batch according to batch siye set in the dataset

    X_fake, y_fake = model.generator.generate(1)

    while X_fake.shape[0] <= X_real.shape[0]:
        X_ , y_ = model.generator.generate(256) # Change this if GPU memory is not enough
        X_fake = torch.cat([X_fake, X_], dim=0)
        y_fake = torch.cat([y_fake, y_], dim=0)

    X_fake = X_fake[:X_real.shape[0]]
    y_fake = y_fake[:y_real.shape[0]]

    X_real, X_fake, y_real, y_fake = to_numpy([X_real, X_fake, y_real, y_fake])

    return X_real, X_fake, y_real, y_fake