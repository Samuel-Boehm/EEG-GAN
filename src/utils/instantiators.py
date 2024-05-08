from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb

from src.utils import pylogger
from src.models import GAN, Generator, Critic, SpectralCritic

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_model(models_cfg: DictConfig, n_samples:int) -> List[Callback]:
    """
    Function to instantiate the GAN model from the configuration file

    Parameters:
    ----------
        cfg (DictConfig)
            configuration for the GAN model

    Returns:
    ----------
        GAN
            GAN model
    """
    models = dict()

    generator:Generator = hydra.utils.instantiate(models_cfg.generator, n_samples=n_samples)
    models['generator'] = generator
    
    critic:Critic = hydra.utils.instantiate(models_cfg.critic, n_samples=n_samples)
    models['critic'] = critic

    if hasattr(models_cfg, 'spectral_critic'):
        spectral_critic:SpectralCritic = hydra.utils.instantiate(models_cfg.spectral_critic, n_samples=n_samples)
        models['spectral_critic'] = spectral_critic
    
    return hydra.utils.instantiate(models_cfg.gan, **models , optimizer=models_cfg.optimizer)

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, conf in callbacks_cfg.items():
        if isinstance(conf, DictConfig) and "_target_" in conf:
            log.info(f"Instantiating callback <{conf._target_}>")
            callbacks.append(hydra.utils.instantiate(conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> Logger:

    for _, conf in logger_cfg.items():
        if isinstance(conf, DictConfig) and "_target_" in conf:
            print(f"Instantiating logger <{conf._target_}>")
            logger = hydra.utils.instantiate(conf)
        else:
            raise TypeError(f"Logger config must be a DictConfig, got {type(conf)}!")
    return logger