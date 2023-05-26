from braindecode.datautil import load_concat_dataset
from torch.utils.data import DataLoader


dataset_path = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/SchirrmeisterChs'


windows_dataset = load_concat_dataset(
    path=dataset_path,
    preload=True,
    target_name=None,
)

df = windows_dataset.get_metadata()

print(len(set(df["target"])))