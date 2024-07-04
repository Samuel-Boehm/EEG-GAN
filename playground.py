from src.data.datamodule import ProgressiveGrowingDataset
from src.visualization.plot import plot_time_domain_by_target, plot_spectrum, compute_stft, plot_stft
import matplotlib.pyplot as plt
import numpy as np
from src.utils.utils import to_numpy

plt.style.use('ggplot')

ds = ProgressiveGrowingDataset('dummy_data', batch_size=200, n_stages=1)

ds.set_stage(1)

dl = ds.train_dataloader()

X, y = next(iter(dl))

X, y = to_numpy([X, y])

n_channels = X.shape[1]

if n_channels <= 3:
        nrows = 1
        ncols = n_channels
else:
    ncols = nrows = int(np.ceil(np.sqrt(n_channels)))

fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))

axs = axs.flat[:n_channels]
for i in range(n_channels):
    plot_time_domain_by_target(X[:, i, :], y,  ax=axs[i], mapping={0: 'class_0', 1: 'class_1'},)

plt.show()

# Plot Spectrum
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# mapping_split