from hydra import compose, initialize
from src.visualization.plot import plot_spectrum_by_target, plot_bin_stats, multi_class_time_domain, mapping_split
import numpy as np

from utils import return_real_and_fake

def main(model_path:str, stage:int) -> None:
    
    # Load config
    with initialize(version_base=None, config_path=str(model_path)):
        cfg = compose(config_name="config")

<<<<<<< HEAD
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
    model.generator.set_stage(stage)

    ds = ProgressiveGrowingDataset(cfg.data.dataset_name, batch_size=512, n_stages=cfg.trainer.scheduler.n_stages)
    ds.set_stage(stage)

    dl = ds.train_dataloader()

    X_real, y_real = next(iter(dl))

    X_fake, y_fake = model.generator.generate(1)

    while X_fake.shape[0] <= X_real.shape[0]:
        X_ , y_ = model.generator.generate(256) # Change this if GPU memory is not enough
        X_fake = torch.cat([X_fake, X_], dim=0)
        y_fake = torch.cat([y_fake, y_], dim=0)

    X_fake = X_fake[:X_real.shape[0]]
    y_fake = y_fake[:y_real.shape[0]]

    X_real, X_fake, y_real, y_fake = to_numpy([X_real, X_fake, y_real, y_fake])
=======
    # Load real data and generate data
    X_real, X_fake, y_real, y_fake = return_real_and_fake(model_path, cfg, stage)
>>>>>>> cf53a381b6af548491bccd9db3412d10972931df

    print(f'X_real shape: {X_real.shape} X_fake shape: {X_fake.shape}')

    base_sfreq = cfg.data.sfreq
    _stage = cfg.trainer.scheduler.n_stages - stage
    current_sfreq = int(base_sfreq // 2**_stage)

    class_mapping = dict(zip(range(len(cfg.data.classes)), cfg.data.classes))

     # Plot Time domain
    
    figures = multi_class_time_domain(X_real, X_fake, y_real, y_fake, class_mapping,
                                      cfg.data.channels)
    
    # Plots bin statistics

    ## split real and fake into labels
    X_real_split = mapping_split(X_real, y_real, class_mapping)
    X_fake_split = mapping_split(X_fake, y_fake, class_mapping)

    ## for each label create a real and fake plot
    for label in class_mapping.values():
        bin_stats, stft_real, stft_fake = plot_bin_stats(X_real_split[label], X_fake_split[label],
                                                        fs=current_sfreq, channels=cfg.data.channels)
        ## add plots to figures
        figures[f'bin_stats_{label}'] = bin_stats
        figures[f'stft_real_{label}'] = stft_real
        figures[f'stft_fake_{label}'] = stft_fake
    
   

    # Plot spectrum
    pooled_data = np.concatenate((X_real, X_fake), axis=0)
    pooled_labels = np.concatenate((np.zeros(X_real.shape[0]), np.ones(X_fake.shape[0])))
    
    _, fig_frequency_domain = plot_spectrum_by_target(pooled_data, pooled_labels,
                                                      cfg.data.sfreq, show_std=True,
                                                      mapping={0: 'real', 1: 'fake'})

    figures['frequency_domain'] = fig_frequency_domain

    for key, fig in figures.items():
        fig.savefig(f'{model_path}/{key}_stage{stage}.png')
    

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    # Setup parser
    parser = argparse.ArgumentParser(description='Make plots from EEG-GAN model')
    parser.add_argument('model_name', type=str, help='Name of model to load')
    parser.add_argument('stage', type=int, help='Stage of model to load')
    args = parser.parse_args()

    model_path = Path(f'trained_models/{args.model_name}')
    
    main(model_path, args.stage, )
    