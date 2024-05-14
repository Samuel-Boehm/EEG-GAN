from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import numpy as np 

import wandb

from src.utils.evaluation_utils import evaluate_model

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    instantiate_model,
)

log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.get("data"), n_stages=cfg.trainer.scheduler.n_stages)

    n_samples = int(cfg.data.sfreq * cfg.data.length_in_seconds)
    log.info(f"Instantiating model <{cfg.model.gan._target_}>")
    model: LightningModule = instantiate_model(cfg.get("model"), n_samples=n_samples)

    log.info("Instantiating training scheduler...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: Logger = instantiate_loggers(cfg.get("logger"))

    max_epochs = int(np.sum(cfg.trainer.scheduler.epochs_per_stage))

    log.info(f"Instantiating training scheduler <{cfg.trainer.scheduler._target_}>")
    scheduler = hydra.utils.instantiate(cfg.trainer.scheduler)
    callbacks.append(scheduler)
    
    log.info(f"Instantiating trainer <{cfg.trainer.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer.trainer, callbacks=callbacks, logger=logger,
                                              reload_dataloaders_every_n_epochs=1, max_epochs=max_epochs)
    
    

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    assert cfg.model.params.n_classes == len(cfg.data.classes), "Number of classes must match!"
    assert cfg.model.params.n_channels == len(cfg.data.channels), "Number of channels must match!"

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, object_dict = train(cfg)

    # evaluate the model
    dm = object_dict["datamodule"]
    dm.set_stage(cfg.trainer.scheduler.n_stages)
    dataloader = dm.train_dataloader()
    
    model = object_dict["model"]

    fig_time_domain, fig_frequency_domain = evaluate_model(model, dataloader, cfg)
        
    wandb.log({"time_domain": wandb.Image(fig_time_domain), "frequency_domain": wandb.Image(fig_frequency_domain)})
   
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

    
    # dataloader = trainer.train_dataloader
    # model = trainer.model
    # Looger
