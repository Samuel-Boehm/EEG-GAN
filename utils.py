from typing import Tuple
from src.utils.utils import to_numpy
import hydra
from src.models.gan import GAN
from src.data.datamodule import ProgressiveGrowingDataset
import torch
from omegaconf import DictConfig
from numpy import ndarray

def generate_from_checkpont(model_path: str, cfg: DictConfig, stage: int, n_examples: int = 1) -> Tuple[ndarray, ndarray]:
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

    model = GAN.load_from_checkpoint(
        checkpoint,
        **models,
        **cfg.model.gan,
        optimizer=cfg.model.optimizer,
        strict=False
    )

    model.eval()
    model.generator.set_stage(stage)

    # Generate fake data only
    X_fake, y_fake = model.generator.generate(n_examples)

    X_fake, y_fake = to_numpy([X_fake, y_fake])
    return X_fake, y_fake