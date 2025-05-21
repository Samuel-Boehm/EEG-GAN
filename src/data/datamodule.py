import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset


class ProgressiveGrowingDataset(LightningDataModule):
    """
    This class is a LightningDataModule that is used to train the GAN in a progressive growing manner.
    It loads metadata, selectively loads EEG trials, applies optional preprocessing,
    and provides DataLoaders based on the specified mode.

    Methods:
    ----------
    load_metadata(): Loads metadata from all .json files.
    load_tensors(classes: Optional[List[str]] = None): Loads .pt tensors based on optional label filtering.
    preprocess(): Applies optional preprocessing to the loaded data.
    setup(stage: str): Sets up the dataset for a given stage (currently resamples).
    train_dataloader(): Returns a DataLoader for the training data (if mode is 'classification').
    val_dataloader(): Returns a DataLoader for the validation data (if mode is 'classification').
    dataloader(): Returns a DataLoader for all data (if mode is 'gan').
    set_stage(stage: int): Reloads and resamples the data for the current stage.
    """

    def __init__(
        self,
        folder_name: str,
        batch_size: int,
        n_stages: int,
        sfreq: int,
        mode: str = "gan",
        classes: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.debug = False
        if folder_name == "debug":
            print("Dataloader entering debug mode, only using dummy data.")
            self.debug = True

        else:
            self.data_dir = Path.cwd() / "datasets" / folder_name
        self.batch_size = batch_size
        self.n_stages = n_stages
        self.base_sfreq = sfreq
        self.mode = mode
        self.classes = classes
        self.metadata = {}
        self.X = None
        self.y = None
        self.split = None
        super().__init__()

    def load_metadata(self):
        self.metadata = {}
        for metadata_path in self.data_dir.rglob("*.json"):
            with open(metadata_path, "r") as f:
                try:
                    data = json.load(f)
                    filename_base = metadata_path.stem
                    self.metadata[filename_base] = data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {metadata_path}: {e}")

    def load_tensors(self, classes: Optional[List[str]] = None):
        self.X = []
        self.y = []
        self.split = []
        print(
            f"Found {len(list(self.data_dir.rglob('*.pt')))} tensors in {self.data_dir}."
        )
        for tensor_path in self.data_dir.rglob("*.pt"):
            filename_base = tensor_path.stem
            if filename_base in self.metadata:
                metadata = self.metadata[filename_base]
                if classes is None or metadata.get("class_name") in classes:
                    tensor = torch.load(tensor_path)
                    self.X.append(tensor)
                    self.y.append(metadata.get("label"))
                    self.split.append(metadata.get("split"))

        if self.X:
            self.X = torch.stack(self.X).float()
            if self.split:
                self.split = np.array(self.split)
        else:
            print("No tensors loaded based on the criteria.")

        # Make y to be consistent between 0 and n_classes
        if self.y:
            self.y = np.array(self.y)
            unique_classes = np.unique(self.y)
            class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
            self.y = np.vectorize(class_mapping.get)(self.y)
        else:
            print("No labels loaded based on the criteria.")

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.debug:
            self.load_metadata()
            self.set_stage(1)  # Initial resampling to the smallest frequency
        else:
            self.set_stage(1)  # Still need to set stage for dummy data

    def train_dataloader(self) -> Optional[DataLoader]:
        if (
            self.mode == "classification"
            and self.X is not None
            and self.y is not None
            and self.split is not None
        ):
            train_dataset = TensorDataset(
                self.X[self.split == "train"], self.y[self.split == "train"]
            )
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        elif self.mode == "classification":
            print(
                "Warning: Train split not available or data not loaded for classification."
            )
            return None
        elif self.mode == "gan" and self.X is not None and self.y is not None:
            return DataLoader(
                TensorDataset(self.X, torch.Tensor(self.y)),
                batch_size=self.batch_size,
                shuffle=True,
            )
        return None

    def val_dataloader(self) -> Optional[DataLoader]:
        if (
            self.mode == "classification"
            and self.X is not None
            and self.y is not None
            and self.split is not None
        ):
            val_dataset = TensorDataset(
                self.X[self.split == "test"], self.y[self.split == "test"]
            )
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        elif self.mode == "classification":
            print(
                "Warning: Test/validation split not available or data not loaded for classification."
            )
            return None
        return None

    def dataloader(self) -> Optional[DataLoader]:
        if self.mode == "gan" and self.X is not None:
            gan_dataset = TensorDataset(self.X)
            return DataLoader(gan_dataset, batch_size=self.batch_size, shuffle=True)
        elif self.mode == "gan":
            print("Warning: Data not loaded for GAN mode.")
            return None
        return None

    def set_stage(self, stage: int):
        stage = self.n_stages - stage  # override external with internal stage variable
        current_sfreq = int(self.base_sfreq // 2**stage)
        if not self.debug:
            self.load_tensors(classes=self.classes)

            # If the data is already at the correct frequency, we don't need to resample it.
            if current_sfreq == self.base_sfreq:
                return

            # Resample the data
            self.X = self.resample(self.X.numpy(), self.base_sfreq, current_sfreq)
            self.X = torch.tensor(self.X).float()

        elif self.debug:
            self.X, self.y = self.generate_fake_eeg(
                n_seconds=2.5,
                sfreq=self.base_sfreq,
                num_channels=21,
                num_trials_per_class=50,
                differing_channels=[3, 7, 8, 9, 18],
            )
            self.X = torch.tensor(self.X).float()
            self.y = torch.tensor(self.y).int()

            if current_sfreq == self.base_sfreq:
                return

            # Resample the data
            self.X = self.resample(self.X.numpy(), self.base_sfreq, current_sfreq)
            self.X = torch.tensor(self.X).float()

    def resample(
        self,
        x: np.ndarray,
        old_sfreq: float,
        new_sfreq: float,
        axis: int = -1,
        npad: int = 100,
        pad_mode: str = "reflect",
        window: str = "boxcar",
    ) -> np.ndarray:
        # Determine target length for the original, unpadded signal
        orig_len = x.shape[axis]
        target_len = int(round(orig_len * new_sfreq / old_sfreq))

        # Pad along the resampling axis if requested
        if npad > 0:
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (npad, npad)
            x = np.pad(x, pad_width, mode=pad_mode)

        # The new length for the padded signal:
        padded_len = x.shape[axis]
        new_padded_len = int(round(padded_len * new_sfreq / old_sfreq))

        # Optionally apply a window to taper the padded edges
        if window != "boxcar":
            win = signal.get_window(window, padded_len)
            # Reshape the window so it can be broadcast along the correct axis
            shape = [1] * x.ndim
            shape[axis] = padded_len
            win = win.reshape(shape)
            x = x * win

        # Resample the padded signal
        x_resampled = signal.resample(x, new_padded_len, axis=axis)

        # Remove the padded segments from the resampled data.
        # Compute the resampled padding length:
        new_npad = int(round(npad * new_sfreq / old_sfreq))
        slicer = [slice(None)] * x.ndim
        slicer[axis] = slice(new_npad, -new_npad)
        x_resampled = x_resampled[tuple(slicer)]

        # Ensure the resampled data has the expected target length.
        if x_resampled.shape[axis] != target_len:
            x_resampled = signal.resample(x_resampled, target_len, axis=axis)

        return x_resampled

    def generate_fake_eeg(
        self,
        n_seconds,
        sfreq,
        num_channels=21,
        num_trials_per_class=50,
        differing_channels=[3, 7, 8, 9, 18],
    ):
        """
        Generate fake EEG data with two classes.

        Returns:
        --------
        X : ndarray
            EEG data with shape (num_trials, num_channels, num_samples)
        y : ndarray
            Class labels (0 or 1)
        """

        X = np.zeros((num_trials_per_class * 2, num_channels, int(n_seconds * sfreq)))
        y = np.zeros((num_trials_per_class * 2,), dtype=int)

        freqs = np.linspace(0.5, sfreq / 2, num_channels)
        time = np.arange(n_seconds * sfreq) / sfreq

        for i in range(num_channels):
            signal = np.sin(2 * np.pi * freqs[i] * time)
            X[:, i, :] = signal

        # Add noise
        noise = np.random.normal(0, 0.5, X.shape)
        X += noise

        # Create class different channels for class 1
        for ch in differing_channels:
            X[num_trials_per_class:, ch, :] = np.sin(2 * np.pi * 2 * time + np.pi / 4)

        y[:num_trials_per_class] = 0
        y[num_trials_per_class:] = 1

        # Shuffle the data
        shuffled_indices = np.random.permutation(X.shape[0])
        X = X[shuffled_indices]
        y = y[shuffled_indices]

        return X, y


if __name__ == "__main__":
    # Example usage
    datamodule = ProgressiveGrowingDataset(
        folder_name="HGD_clinical_normalized",
        batch_size=32,
        n_stages=5,
        sfreq=256,
        mode="classification",
        classes=["rest", "right_hand"],
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    # val_loader = datamodule.val_dataloader()
    # gan_loader = datamodule.dataloader()

    for batch in train_loader:
        print(batch)
