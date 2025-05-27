from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import hydra
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    instantiate_model,
)

log = RankedLogger(__name__, rank_zero_only=True)


def print_module_parameters(module: nn.Module, top_n: int = 5):
    """
    Print parameters for the largest individual layers in a neural network.

    Args:
        module: The PyTorch nn.Module to analyze
        top_n: Number of largest layers to display
    """
    print(f"Inspecting parameters for module: {module.__class__.__name__}\n{'-' * 40}")

    # Track layers by their type and parameter count
    layers_by_type = defaultdict(list)
    total_params = 0

    # Helper function to recursively analyze modules
    def analyze_module(mod, prefix=""):
        nonlocal total_params

        for name, child in mod.named_children():
            child_path = f"{prefix}.{name}" if prefix else name

            # If it's a leaf module (a layer), count its parameters
            if len(list(child.children())) == 0:
                param_count = sum(
                    p.numel() for p in child.parameters() if p.requires_grad
                )
                total_params += param_count

                if param_count > 0:  # Only track modules with parameters
                    layer_type = child.__class__.__name__
                    layers_by_type[layer_type].append((child_path, child, param_count))
            else:
                # Recursively analyze child modules
                analyze_module(child, child_path)

    # Start recursive analysis
    analyze_module(module)

    # Find the largest individual layers across all types
    all_layers = []
    for layer_type, layers in layers_by_type.items():
        all_layers.extend(layers)

    largest_layers = sorted(all_layers, key=lambda x: x[2], reverse=True)[:top_n]

    # Print results
    print(f"{'-' * 40}")
    print(f"Total Parameters: {total_params:,}")
    print(f"{'-' * 40}")

    print(f"Largest {top_n} Individual Layers:")
    for i, (path, layer, param_count) in enumerate(largest_layers, 1):
        print(f"{i}. {layer.__class__.__name__} at {path}")
        print(
            f"   Parameters: {param_count:,} ({param_count / total_params * 100:.2f}% of total)"
        )

    # Print parameter distribution by layer type
    print(f"\n{'-' * 40}")
    print("Parameter Distribution by Layer Type:")

    type_totals = {}
    for layer_type, layers in layers_by_type.items():
        type_total = sum(param_count for _, _, param_count in layers)
        type_totals[layer_type] = (type_total, len(layers))

    for layer_type, (type_total, count) in sorted(
        type_totals.items(), key=lambda x: x[1][0], reverse=True
    )[:10]:
        print(
            f"{layer_type}: {type_total:,} parameters across {count} layers "
            + f"({type_total / total_params * 100:.2f}% of total)"
        )


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    n_samples = int(cfg.data.sfreq * (cfg.data.tmax - cfg.data.tmin))
    cfg.model.generator._target_ = "src.models.components.generators.upsample.Generator"
    model: LightningModule = instantiate_model(
        models_cfg=cfg.get("model"), n_samples=n_samples
    )

    print("Generator Parameters:")
    print_module_parameters(model.generator)

    print("Critic Parameters:")
    print_module_parameters(model.critic)


@hydra.main(version_base="1.3", config_path="../configs", config_name="debug.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    assert cfg.model.params.n_classes == len(cfg.data.classes), (
        "Number of classes must match!"
    )
    assert cfg.model.params.n_channels == len(cfg.data.channels), (
        "Number of channels must match!"
    )

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
