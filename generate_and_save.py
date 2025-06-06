import os
from pathlib import Path
import torch
import numpy as np
import json
from omegaconf import OmegaConf
import hydra
import yaml

from src.utils import instantiate_model  # adjust as needed
from src.utils.evaluation_utils import to_numpy

# --- 1. Configurable parameters ---
RUN_ID = "run-20250602_140759-e56pbtba"  # Set this to your run ID as needed
N_SAMPLES = 1 # Debug mode: generate only 100 samples. Change to 8000 for full generation.
GENERATED_DATASET_DIR = Path("datasets/generated") / RUN_ID

RUN_DIR = f"wandb/{RUN_ID}"
CONFIG_PATH = f"{RUN_DIR}/files/.hydra/config.yaml"
ckpt_dir = Path(RUN_DIR) / "files"
ckpt_files = list(ckpt_dir.glob("*.ckpt"))
if not ckpt_files:
    raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
CKPT_PATH = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]

GENERATED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Load config ---
cfg = OmegaConf.load(CONFIG_PATH)

# --- 3. Instantiate model ---
n_samples_per_trial = int(cfg.data.sfreq * (cfg.data.tmax - cfg.data.tmin))
model = instantiate_model(models_cfg=cfg.model, n_samples=n_samples_per_trial)

# --- 4. Load checkpoint ---
state_dict = torch.load(CKPT_PATH, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.eval()

# --- 4.1 Set stages as in your loading script ---
model.current_stage = cfg.trainer.scheduler.n_stages
model.generator.set_stage(model.current_stage)
model.critic.set_stage(model.current_stage)
model.sp_critic.set_stage(model.current_stage)

# --- 5. Generate fake samples ---
# Get number of classes
class_names = list(cfg.data.classes)
num_classes = len(class_names)
channels = list(cfg.data.channels)

# Optionally, generate balanced classes
samples_per_class = N_SAMPLES // num_classes
remainder = N_SAMPLES % num_classes

X_fake_list = []
y_fake_list = []
split_list = []

for class_idx, class_name in enumerate(class_names):
    n_gen = samples_per_class + (1 if class_idx < remainder else 0)
    # Assume the model.generator.generate(n) returns (X_fake, y_fake)
    # If your generator needs class labels (conditional GAN), pass them as needed
    X_fake, y_fake = model.generator.generate(n_gen)
    X_fake, y_fake = to_numpy((X_fake, y_fake))
    X_fake_list.append(X_fake)
    y_fake_list.append(y_fake)

# Concatenate all generated samples
X_fake = np.concatenate(X_fake_list, axis=0)
y_fake = np.concatenate(y_fake_list, axis=0)

# --- 6. Shuffle and split into train/test (80/20) ---
num_samples = X_fake.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

split_point = int(0.8 * num_samples)
train_indices = indices[:split_point]
test_indices = indices[split_point:]

splits = np.array(["train"] * len(train_indices) + ["test"] * len(test_indices), dtype=object)
split_indices = np.concatenate([train_indices, test_indices])
X_fake = X_fake[split_indices]
y_fake = y_fake[split_indices]

# --- 7. Save samples in the same format as real data ---
# Prepare reverse mapping for class names
mapping = dict(zip(range(len(cfg.data.classes)), cfg.data.classes))

for i in range(X_fake.shape[0]):
    filename_base = f"000_fake_{i:04d}"
    tensor_filename = filename_base + ".pt"
    metadata_filename = filename_base + ".json"
    split = splits[i]
    label = int(y_fake[i])
    class_name = mapping[label]

    # Save tensor
    torch.save(torch.tensor(X_fake[i]).float(), GENERATED_DATASET_DIR / tensor_filename)

    # Save metadata
    metadata = {
        "class_name": class_name,
        "split": split,
        "subject": 1,  # fake data, always 1
        "label": label,
    }
    with open(GENERATED_DATASET_DIR / metadata_filename, "w") as f:
        json.dump(metadata, f)

# --- 8. Save the config file for the generated dataset ---

# Compose config dict for saving
config_dict = {
    "_target_": "src.data.datamodule.ProgressiveGrowingDataset",
    "folder_name": f"generated/{RUN_ID}",  # relative to /datasets
    "dataset_name": f"fake-{RUN_ID}",
    "subject_id": [1],
    "channels": list(cfg.data.channels),
    "sfreq": cfg.data.sfreq,
    "tmin": cfg.data.tmin,
    "tmax": cfg.data.tmax,
    "batch_size": cfg.data.batch_size,
    "classes": list(cfg.data.classes),
}

config_save_path = GENERATED_DATASET_DIR / "config.yaml"
with open(config_save_path, "w") as f:
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

print(f"Saved config file to {config_save_path}")
print(f"Generation complete. {X_fake.shape[0]} samples saved to {GENERATED_DATASET_DIR}")
