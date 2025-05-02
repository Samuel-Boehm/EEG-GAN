from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import signal

# Configuration
# ---------
# You can either set the data_dir and config_path directly, or set the folder_name.  Setting the folder_name is preferred.
folder_name = "32_zca"
data_dir: Path = None  # If folder_name is set, this will be set automatically.
config_path: Path = None  # If folder_name is set, this will be set automatically.
if folder_name is not None:
    data_dir = Path.cwd() / "datasets" / folder_name
    config_path = data_dir / "config.yaml"

# If you want to specify the data_dir and config_path directly, uncomment these lines and modify them.
# data_dir = Path.cwd() / 'datasets' / 'your_dataset_folder'  # Replace with your actual data directory
# config_path = data_dir / 'config.yaml'  # Replace with your actual config file path

plot_dir = Path.cwd() / "plots" / folder_name  # Directory to save the plots
plot_dir.mkdir(exist_ok=True)

# Helper Functions
# --------


def load_data(data_dir: Path, config_path: Path) -> Dict:
    """Loads the EEG data and configuration.

    Args:
        data_dir (Path): Path to the directory containing the data files.
        config_path (Path): Path to the configuration file.

    Returns:
        Dict: A dictionary containing the EEG data, labels, splits, and configuration.
              The keys are 'X', 'y', 'split', and 'config'.
              'X' is a torch.Tensor of shape (n_trials, n_channels, n_samples).
              'y' is a list of labels (strings).
              'split' is a numpy array of strings ('train' or 'test').
              'config' is a dictionary containing the configuration from the YAML file.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    X = []
    y = []
    split = []
    for trial_file in data_dir.glob("*.pt"):
        trial_data = torch.load(trial_file, weights_only=True)
        X.append(trial_data)
        # Extract label and split from the corresponding JSON file
        metadata_file = data_dir / (trial_file.stem + ".json")
        if metadata_file.exists():
            with open(metadata_file, "r") as mf:
                metadata = yaml.safe_load(mf)  # Use yaml.safe_load for JSON too
                y.append(metadata["class_name"])
                split.append(metadata["split"])
        else:
            print(f"Warning: Metadata file not found for {trial_file}. Skipping.")
            y.append("unknown")  # Add a placeholder if metadata is missing
            split.append("unknown")

    X = torch.stack(X).numpy()  # Convert list of tensors to a single tensor
    return {"X": X, "y": y, "split": np.array(split), "config": config}


def plot_all_classes(data: Dict, plot_dir: Path) -> None:
    """Plots the time and frequency domain for each class in a single plot,
    and also plots each single channel.

    Args:
        data (Dict): A dictionary containing the EEG data, labels, splits, and configuration.
        plot_dir (Path): Directory to save the plots.
    """
    X, y, split, config = data["X"], data["y"], data["split"], data["config"]
    classes = config["classes"]
    sfreq = config["sfreq"]
    channels = config["channels"]  # Get the channel names
    t_min = config["tmin"]  # Get t_min from config
    t_max = config["tmax"]  # Get t_max from config

    # Define colors for each class
    colors = ["b", "g", "r", "c", "m", "y", "k"]  # Add more colors if needed
    if len(classes) > len(colors):
        raise ValueError(
            f"Number of classes ({len(classes)}) exceeds number of available colors ({len(colors)})"
        )

    # Create the figure and axes for the combined class plot
    fig_class, axs_class = plt.subplots(2, 1, figsize=(10, 8))

    for i, class_name in enumerate(classes):
        # Get data for the current class
        class_indices = [j for j, label in enumerate(y) if label == class_name]
        eeg_data_class = X[class_indices]  # Shape: (n_trials, n_channels, n_samples)
        eeg_data_class_mean = eeg_data_class.mean(
            axis=0
        )  # Shape: (n_channels, n_samples)

        # Plot the average across channels for the current class, in the combined plot
        time = np.linspace(t_min, t_max, eeg_data_class_mean.shape[1])
        frequencies, power_spectral_density = signal.welch(
            eeg_data_class_mean, sfreq, axis=1
        )

        axs_class[0].plot(
            time, eeg_data_class_mean.mean(axis=0), label=class_name, color=colors[i]
        )
        axs_class[1].plot(
            frequencies,
            power_spectral_density.mean(axis=0),
            label=class_name,
            color=colors[i],
        )

    # Add labels and title to the combined class plot
    axs_class[0].set_title("Time Domain - All Classes")
    axs_class[0].set_xlabel("Time (s)")
    axs_class[0].set_ylabel("Amplitude (µV)")
    axs_class[0].grid(True)
    if t_min is not None and t_max is not None:
        axs_class[0].set_xlim(t_min, t_max)
    axs_class[0].legend()

    axs_class[1].set_title("Frequency Domain - All Classes")
    axs_class[1].set_xlabel("Frequency (Hz)")
    axs_class[1].set_ylabel("Power Spectral Density (µV^2/Hz)")
    axs_class[1].grid(True)
    axs_class[1].legend()

    fig_class.suptitle("Time and Frequency Domain for All Classes")
    plt.tight_layout()
    plt.savefig(plot_dir / "all_classes.png")
    plt.close(fig_class)  # Close the figure

    # Plot each channel separately for each class
    channel_dir = plot_dir / "single_channels"
    channel_dir.mkdir(exist_ok=True)
    for i, channel_name in enumerate(channels):
        # Create a new figure for each channel
        fig_channel, axs_channel = plt.subplots(2, 1, figsize=(10, 8))
        for j, class_name in enumerate(classes):
            class_indices = [k for k, label in enumerate(y) if label == class_name]
            eeg_data_class = X[class_indices]
            eeg_data_channel = eeg_data_class[:, i, :]  # Shape: (n_trials, n_samples)
            time = np.linspace(t_min, t_max, eeg_data_channel.shape[1])
            frequencies, power_spectral_density = signal.welch(
                eeg_data_channel, sfreq, axis=1
            )

            axs_channel[0].plot(
                time, eeg_data_channel.mean(axis=0), label=class_name, color=colors[j]
            )
            axs_channel[1].plot(
                frequencies,
                power_spectral_density.mean(axis=0),
                label=class_name,
                color=colors[j],
            )

        axs_channel[0].set_title(f"Time Domain - Channel: {channel_name}")
        axs_channel[0].set_xlabel("Time (s)")
        axs_channel[0].set_ylabel("Amplitude (µV)")
        axs_channel[0].grid(True)
        if t_min is not None and t_max is not None:
            axs_channel[0].set_xlim(t_min, t_max)
        axs_channel[0].legend()

        axs_channel[1].set_title(f"Frequency Domain - Channel: {channel_name}")
        axs_channel[1].set_xlabel("Frequency (Hz)")
        axs_channel[1].set_ylabel("Power Spectral Density (µV^2/Hz)")
        axs_channel[1].grid(True)
        axs_channel[1].legend()

        fig_channel.suptitle(f"Time and Frequency Domain for Channel: {channel_name}")
        plt.tight_layout()
        plt.savefig(channel_dir / f"{channel_name}.png")
        plt.close(fig_channel)  # Close the figure


# Main
# ----
if __name__ == "__main__":
    # Load the data
    data = load_data(data_dir, config_path)

    # Plot the data
    plot_all_classes(data, plot_dir)
    print(f"Plots saved to {plot_dir}")
