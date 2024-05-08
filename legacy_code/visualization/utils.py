import matplotlib.pyplot as plt
import numpy as np


def labeled_tube_plot(x, data_y, tube_y, labels,
                      title="", xlabel="", ylabel="", axes=None,  autoscale = True):
    x = np.asarray(x)
    data_y = np.asarray(data_y)
    tube_y = np.asarray(tube_y)

    if axes is None:
        figure, axes = plt.subplots()

    colors = []
    for i, label in enumerate(labels):
        y_tmp = data_y[i]
        tube_tmp = tube_y[i]
        p = axes.fill_between(x, y_tmp + tube_tmp, y_tmp - tube_tmp, alpha=0.5, label=labels[i])
        colors.append(p._original_facecolor)

    for i, label in enumerate(labels):
        axes.plot(x, data_y[i], lw=2, color=colors[i])

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    if autoscale:
        axes.set_ylabel(ylabel)
        axes.set_xlim(x.min(), x.max())
        axes.set_ylim(np.nanmin(data_y - tube_y), np.nanmax(data_y + tube_y))
    axes.legend()

    return figure, axes



