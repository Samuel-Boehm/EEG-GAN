from braindecode.models import deep4
from omegaconf import DictConfig
import mne
from braindecode.datasets import create_from_X_y
from utils import return_real_and_fake
from hydra import compose, initialize
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit
from braindecode.models.deep4 import Deep4Net
import torch

def main(model_path:str, stage:int) -> None:
    # Load config
    with initialize(version_base=None, config_path=str(model_path)):
        cfg = compose(config_name="config")

    base_sfreq = cfg.data.sfreq
    _stage = cfg.trainer.scheduler.n_stages - stage
    current_sfreq = int(base_sfreq // 2**_stage)

    # Load real data and generate data
    X_real, X_fake, y_real, y_fake = return_real_and_fake(model_path, cfg, stage)

    ch_names = list(cfg.data.channels)

    ds_real = create_from_X_y(
    X_real, y_real, drop_last_window=False, sfreq=current_sfreq,
    ch_names=ch_names,)

    ds_fake = create_from_X_y(
    X_fake, y_fake, drop_last_window=False, sfreq=current_sfreq,
    ch_names=ch_names,)

    x_i, y_i, _ = ds_real[0]


    model = Deep4Net(
    n_chans=x_i.shape[0],
    n_classes=2,
    input_window_samples=x_i.shape[1],
    final_conv_length='auto',
    )

    # Send model to GPU
    if torch.cuda.is_available():
        model = model.cuda()


    clf = EEGClassifier(
        module=model)

    # Train model
    clf.fit(ds_fake, y=y_fake, epochs=300)


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