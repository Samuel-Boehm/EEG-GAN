import hydra
from pathlib import Path
from omegaconf import DictConfig
import omegaconf
from src.data.preprocess import preprocess_moabb

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    cfg = cfg.data

    base_dir = Path(__file__).parent.parent
    dataset_dir = Path(base_dir, "datasets", cfg.folder_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    label_map = preprocess_moabb(**cfg, dataset_dir=dataset_dir)
    
    # add label map to config - there is no existing field fo it:
    # Disable strict mode to allow new keys.
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.label_map = label_map

    config_path = Path(dataset_dir, "config.yaml")
    with open(config_path, 'w') as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()