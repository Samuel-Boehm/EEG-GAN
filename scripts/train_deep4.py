import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig

import wandb
from src.classifier.deep4 import Deep4LightningModule
from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the Deep4 model for classification.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.get("data"), n_stages=1, mode="classification"
    )

    # Figure out the input shape
    n_classes = cfg.model.n_classes
    n_chans = cfg.model.n_channels

    log.info("Instantiating Deep4 model for classification")
    model = Deep4LightningModule(
        input_window_samples=int(cfg.data.sfreq * (cfg.data.tmax - cfg.data.tmin)),
        n_channels=n_chans,
        n_classes=n_classes,
    )

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: Logger = instantiate_loggers(cfg.get("logger"))

    max_epochs = 100

    log.info("Instantiating trainer")
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
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
        # The checkpoint callback may not exist if not set up, so we use current weights if absent
        ckpt_path = getattr(
            getattr(trainer, "checkpoint_callback", None), "best_model_path", ""
        )
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
    """Main entry point for Deep4 classification training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # Access the global wandb 'run'
    global run
    if run:
        run.name = cfg.run_name

    assert cfg.model.n_classes == len(cfg.data.classes), "Number of classes must match!"
    assert cfg.model.n_channels == len(cfg.data.channels), (
        "Number of channels must match!"
    )

    # Apply extra utilities
    extras(cfg)

    # Train the model
    train(cfg)

    # Rename the run
    run.name = cfg.run_name

    wandb.finish()

    log.info("Finished Training")
    sys.exit("Finished Training")


if __name__ == "__main__":
    run = wandb.init(
        project="EEG-GAN",
    )
    sys.argv.append(f"hydra.run.dir={wandb.run.dir}")
    print(f"Run dir: {wandb.run.dir}")

    main()
