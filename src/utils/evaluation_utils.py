import torch
from torch.utils.data import DataLoader

import numpy as np

from omegaconf import DictConfig
import hydra

import matplotlib.pyplot as plt

from src.models import GAN
from src.visualization.plot import plot_time_domain_by_target, plot_spectrum_by_target
from src.utils.utils import to_numpy

from src.utils import instantiate_model

from src.data.datamodule import ProgressiveGrowingDataset

def evaluate_model(model:GAN, dataloader:DataLoader, cfg:DictConfig):
    
    n_samples = 24
    
    model.eval()
    with torch.no_grad():
        
        X_real = []
        y_real = []
        
        data_iter = iter(dataloader)
        samples_collected = 0

        while samples_collected < n_samples:
            try:
                X_, y_ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                X_, y_ = next(data_iter)

            X_real.append(X_)
            y_real.append(y_)

            samples_collected += len(X_real)

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

    fig_time_domain, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
    axs = axs.flat[:n_channels]

    mapping =dict(zip(range(len(cfg.data.classes)), cfg.data.classes))

    for i, ax in enumerate(axs):
        data = X_real[:, i, :]
        labels = y_real

        plot_time_domain_by_target(data, labels, show_std=True, ax=ax, mapping=mapping, title=channels[i])

    _, fig_frequency_domain = plot_spectrum_by_target(data, labels, cfg.data.sfreq, show_std=True, mapping=mapping)

    return fig_time_domain, fig_frequency_domain
    
    
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
