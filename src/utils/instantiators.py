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


def instantiate_model(models_cfg: DictConfig) -> List[Callback]:
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

    generator = Generator(**models_cfg.generator)
    models['generator'] = generator
    
    critic = Critic(**models_cfg.critic)
    models['critic'] = critic

    if hasattr(models_cfg, 'spectral_critic'):
        spectral_critic = SpectralCritic(**models_cfg.spectral_critic)
        models['spectral_critic'] = spectral_critic
    
    return GAN(**models, **models_cfg.gan, optimizer=models_cfg.optimizer)

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

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(cfg: DictConfig) -> List[Logger]:

    logger: List[Logger] = []

    if not hasattr(cfg, 'logger'):
        print("No logger configuration found!")
        return
    
    run_name = '/'.join(HydraConfig.get()['run']['dir'].split('/')[-2:])

    for _, cb_conf in cfg.logger.items():
        cb_conf['name'] = run_name
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"Instantiating logger <{cb_conf._target_}>")
            logger.append(hydra.utils.instantiate(cb_conf))

    return logger