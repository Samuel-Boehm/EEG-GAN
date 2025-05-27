import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import wandb
from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_model,
    log_hyperparameters,
)
from src.utils.evaluation_utils import evaluate_model

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
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.get("data"), n_stages=cfg.trainer.scheduler.n_stages
    )

    n_samples = int(cfg.data.sfreq * (cfg.data.tmax - cfg.data.tmin))
    log.info(f"Instantiating model <{cfg.model.gan._target_}>")
    model: LightningModule = instantiate_model(
        models_cfg=cfg.get("model"), n_samples=n_samples
    )

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: Logger = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating training scheduler <{cfg.trainer.scheduler._target_}>")
    scheduler = hydra.utils.instantiate(cfg.trainer.scheduler)
    callbacks.append(scheduler)

    max_epochs = int(scheduler.max_epochs)

    log.info(f"Instantiating trainer <{cfg.trainer.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer.trainer,
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=1,
        max_epochs=max_epochs,
    )

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    assert cfg.model.params.n_classes == len(cfg.data.classes), (
        "Number of classes must match!"
    )
    assert cfg.model.params.n_channels == len(cfg.data.channels), (
        "Number of channels must match!"
    )

    # Apply extra utilities
    extras(cfg)

    # Train the model
    metric_dict, object_dict = train(cfg)

    # Evaluate the model
    dm = object_dict["datamodule"]
    dm.set_stage(cfg.trainer.scheduler.n_stages)
    dataloader = dm.train_dataloader()

    model = object_dict["model"]

    figures = evaluate_model(model, dataloader, cfg)

    for key, fig in figures.items():
        wandb.log({key: wandb.Image(fig)})

    # End of training
    wandb.finish()

    log.info("Finished Training")
    sys.exit("Finished Training")


if __name__ == "__main__":
    wandb.init(
        project="EEG-GAN",
    )
    sys.argv.append(f"hydra.run.dir={wandb.run.dir}")
    print(f"Run dir: {wandb.run.dir}")

    main()
