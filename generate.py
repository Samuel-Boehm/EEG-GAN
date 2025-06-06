from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.utils import instantiate_model  # adjust as needed
from src.utils.evaluation_utils import evaluate_model

# --- 1. Set up paths ---
RUN_DIR = "wandb/run-20250602_140759-e56pbtba"
CONFIG_PATH = f"{RUN_DIR}/files/.hydra/config.yaml"
ckpt_dir = Path(RUN_DIR) / "files"
ckpt_files = list(ckpt_dir.glob("*.ckpt"))
if not ckpt_files:
    raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
CKPT_PATH = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]

MEDIA_DIR = Path(RUN_DIR) / "files" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Load config ---
cfg = OmegaConf.load(CONFIG_PATH)

# --- 3. Instantiate datamodule and model ---
datamodule = hydra.utils.instantiate(cfg.data, n_stages=cfg.trainer.scheduler.n_stages)
datamodule.setup()
n_samples = int(cfg.data.sfreq * (cfg.data.tmax - cfg.data.tmin))
model = instantiate_model(models_cfg=cfg.model, n_samples=n_samples)

# --- 4. Load checkpoint ---
state_dict = torch.load(CKPT_PATH, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.eval()

# 4.1 set stages:
datamodule.set_stage(cfg.trainer.scheduler.n_stages)
model.current_stage = cfg.trainer.scheduler.n_stages
model.generator.set_stage(model.current_stage)
model.critic.set_stage(model.current_stage)
model.sp_critic.set_stage(model.current_stage)


# 4.2 Very important to set the alpha parameter to > 1, else we will mix stages
model.generator.alpha = 2
model.critic.alpha = 2 
model.sp_critic.alpha = 2

# --- 5. Get dataloader ---
dataloader = datamodule.train_dataloader()

# --- 6. Evaluate and save figures ---
figures = evaluate_model(model, dataloader, cfg)
for key, fig in figures.items():
    fig_path = MEDIA_DIR / f"{key}.png"
    fig.savefig(fig_path)
    print(f"Saved {fig_path}")
    fig.clf()  # release memory if running interactively

