from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.models import GAN
from src.utils import instantiate_model
from src.utils.utils import to_numpy
from src.visualization.plot import (
    mapping_split,
    plot_spectrum_by_target,
    plot_time_domain,
)


def evaluate_model(
    model: GAN, dataloader: DataLoader, cfg: DictConfig
) -> Dict[str, plt.Figure]:
    n_samples = 280
    batch_size = dataloader.batch_size

    model.eval()
    with torch.no_grad():
        X_real = []
        y_real = []

        data_iter = iter(dataloader)
        samples_collected = 0

        for _ in range(((n_samples // batch_size + 1) * batch_size) // batch_size):
            try:
                X_, y_ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                X_, y_ = next(data_iter)

            X_real.append(X_)
            y_real.append(y_)

        X_real = torch.cat(X_real, dim=0)
        y_real = torch.cat(y_real, dim=0)

        X_fake, y_fake = model.generator.generate(X_real.shape[0])

        X_real, X_fake, y_real, y_fake = to_numpy((X_real, X_fake, y_real, y_fake))

    n_channels = len(cfg.data.channels)
    channels = cfg.data.channels

    if n_channels <= 3:
        nrows = 1
        ncols = n_channels
    else:
        ncols = nrows = int(np.ceil(np.sqrt(n_channels)))

    mapping = dict(zip(range(len(cfg.data.classes)), cfg.data.classes))

    # For each class plot real and fake

    real_dict = mapping_split(X_real, y_real, mapping)
    fake_dict = mapping_split(X_fake, y_fake, mapping)

    out_dict = {}

    for key in cfg.data.classes:
        fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
        axs = axs.flat[:n_channels]
        for i in range(n_channels):
            plot_time_domain(
                real_dict[key][:, i, :],
                ax=axs[i],
                title=channels[i],
                show_std=True,
                label="real",
            )
            plot_time_domain(
                fake_dict[key][:, i, :],
                ax=axs[i],
                title=channels[i],
                show_std=True,
                label="fake",
            )

        out_dict[f"time_domain_{key}"] = fig

    pooled_data = np.concatenate((X_real, X_fake), axis=0)
    pooled_labels = np.concatenate(
        (np.zeros(X_real.shape[0]), np.ones(X_fake.shape[0]))
    )

    _, fig_frequency_domain = plot_spectrum_by_target(
        pooled_data,
        pooled_labels,
        cfg.data.sfreq,
        show_std=True,
        mapping={0: "real", 1: "fake"},
    )

    out_dict["frequency_domain"] = fig_frequency_domain

    return out_dict


if __name__ == "__main__":
    results_dir = (
        "/home/samuelboehm/PhD/EEG-GAN/trained_models/Auswerten/nxgnjfwe/files"
    )

    # Load YAML configuration file
    hydra.initialize_config_dir(config_dir=results_dir)
    cfg = hydra.compose(config_name="config")

    n_samples = int(cfg.data.sfreq * 2.5)

    model: GAN = instantiate_model(cfg.model, n_samples=n_samples)
    checkpoint = torch.load(
        "/home/samuelboehm/PhD/EEG-GAN/trained_models/Auswerten/nxgnjfwe/final.ckpt"
    )

    # Extract the state_dict (contains the weights)
    state_dict = checkpoint.get(
        "state_dict", checkpoint
    )  # Handle different checkpoint formats

    # Load the state_dict into the instantiated model
    model.load_state_dict(state_dict)
    print("Weights loaded successfully!")

    # Set the model to evaluation mode if you're not training
    model.eval()

    dm: LightningDataModule = hydra.utils.instantiate(
        cfg.get("data"), n_stages=cfg.trainer.scheduler.n_stages
    )
    dm.setup()
    dm.set_stage(cfg.trainer.scheduler.n_stages)

    for stage in range(1, cfg.model.params.n_stages + 1):
        model.current_stage = stage
        print(f"Stage: {stage}")
        dm.set_stage(model.current_stage)
        model.generator.set_stage(model.current_stage)
        model.critic.set_stage(model.current_stage)
        model.sp_critic.set_stage(model.current_stage)

        dataloader = dm.train_dataloader()
        out = evaluate_model(model, dataloader, cfg)
        for key, fig in out.items():
            fig.savefig(f"{results_dir}/{key}_{stage}.png", dpi=300)
            plt.close(fig)
