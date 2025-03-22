from braindecode.models import deep4
from omegaconf import DictConfig
import mne
from utils import generate_from_checkpont
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
    X_real, X_fake, y_real, y_fake = generate_from_checkpont(model_path, cfg, stage)

