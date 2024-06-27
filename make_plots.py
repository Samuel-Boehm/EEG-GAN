from hydra import compose, initialize
import hydra
from src.models.gan import GAN
from legacy_code.visualization.stft_plots import plot_bin_stats
from src.data.datamodule import ProgressiveGrowingDataset
import torch
from src.utils.utils import to_numpy
from src.utils.evaluation_utils import evaluate_model


def main(model_path:str, stage:int) -> None:
    
    checkpoint = list(model_path.glob('*.ckpt'))[0]

    with initialize(version_base=None, config_path=str(model_path)):
        cfg = compose(config_name="config")

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

    ds = ProgressiveGrowingDataset('clinical', batch_size=512, n_stages=cfg.trainer.scheduler.n_stages)
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

    print(f'X_real shape: {X_real.shape} X_fake shape: {X_fake.shape}')

    base_sfreq = cfg.data.sfreq
    _stage = cfg.trainer.scheduler.n_stages - stage
    current_sfreq = int(base_sfreq // 2**_stage)

    bin_stats, stft_real, stft_fake = plot_bin_stats(X_real, X_fake, fs=current_sfreq, channels=cfg.data.channels)

    figures = evaluate_model(model, dl, cfg)

    figures['bin_stats'] = bin_stats
    figures['stft_real'] = stft_real
    figures['stft_fake'] = stft_fake

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
    