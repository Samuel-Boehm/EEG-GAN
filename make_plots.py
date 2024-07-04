from hydra import compose, initialize
from src.visualization.plot import plot_spectrum_by_target, plot_bin_stats, multi_class_time_domain, mapping_split
import numpy as np

from utils import return_real_and_fake

def main(model_path:str, stage:int) -> None:
    
    # Load config
    with initialize(version_base=None, config_path=str(model_path)):
        cfg = compose(config_name="config")

    # Load real data and generate data
    X_real, X_fake, y_real, y_fake = return_real_and_fake(model_path, cfg, stage)

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
    