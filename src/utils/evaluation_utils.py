import torch
from torch.utils.data import DataLoader

import numpy as np
from typing import Dict

from omegaconf import DictConfig
import hydra

import matplotlib.pyplot as plt

from src.models import GAN
from src.visualization.plot import plot_time_domain_by_target, plot_spectrum_by_target
from src.utils.utils import to_numpy

from src.utils import instantiate_model

from src.data.datamodule import ProgressiveGrowingDataset

def evaluate_model(model:GAN, dataloader:DataLoader, cfg:DictConfig) -> Dict[str, plt.Figure]:
    
    n_samples = 512
    batch_size = dataloader.batch_size
    
    model.eval()
    with torch.no_grad():
        X_real = []
        y_real = []
        
        data_iter = iter(dataloader)
        samples_collected = 0

        for _ in range(((n_samples // batch_size + 1) * batch_size)//batch_size):
            try:
                X_, y_ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                X_, y_ = next(data_iter)

            X_real.append(X_)
            y_real.append(y_)

        X_real = torch.cat(X_real, dim=0)
        y_real = torch.cat(y_real, dim=0)

        X_fake, y_fake = model.generator.generate(X_real.shape)

        X_real, X_fake, y_real, y_fake = to_numpy((X_real, X_fake, y_real, y_fake))

    n_channels = len(cfg.data.channels)
    channels = cfg.data.channels

    if n_channels <= 3:
        nrows = 1
        ncols = n_channels
    else:
        ncols = nrows = int(np.ceil(np.sqrt(n_channels)))

    fig_td_real, axs_td_real = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
    fig_td_fake, axs_td_fake = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))

    axs_td_real = axs_td_real.flat[:n_channels]
    axs_td_fake = axs_td_fake.flat[:n_channels]

    mapping =dict(zip(range(len(cfg.data.classes)), cfg.data.classes))

    for i in range(n_channels):
        data_real = X_real[:, i, :]
        data_fake = X_fake[:, i, :]

        plot_time_domain_by_target(data_real, y_real, show_std=True, ax=axs_td_real[i], mapping=mapping, title=channels[i])
        plot_time_domain_by_target(data_fake, y_fake, show_std=True, ax=axs_td_fake[i], mapping=mapping, title=channels[i])

    
    pooled_data = np.concatenate((X_real, X_fake), axis=0)
    pooled_labels = np.concatenate((np.zeros(X_real.shape[0]), np.ones(X_fake.shape[0])))

    _, fig_frequency_domain = plot_spectrum_by_target(pooled_data, pooled_labels, cfg.data.sfreq, show_std=True, mapping={0: 'real', 1: 'fake'})

    figures = {'time_domain_real': fig_td_real, 'time_domain_fake': fig_td_fake, 'frequency_domain': fig_frequency_domain}

    return figures
    
    
if __name__ == "__main__":
    
    results_dir = '/home/samuelboehm/EEG-GAN/trained_models/2024-05-08-16-55-57'

    # Load YAML configuration file
    hydra.initialize_config_dir(config_dir=results_dir)
    cfg = hydra.compose(config_name='config')

    n_samples = int(cfg.data.sfreq * cfg.data.length_in_seconds)
    model = instantiate_model(cfg.model, n_samples=n_samples)

    dm:ProgressiveGrowingDataset = hydra.utils.instantiate(cfg.get("data"), n_stages=cfg.callbacks.scheduler.n_stages)
    dm.set_stage(cfg.callbacks.scheduler.n_stages)
    dataloader = dm.train_dataloader()

    fig1, fig2 = evaluate_model(model, dataloader, cfg) 
    
    
    import io
    from PIL import Image

    # Convert the plots to PIL Images
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    img1 = Image.open(buf1)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    img2 = Image.open(buf2)
    
    img1.show(title="My Image")
