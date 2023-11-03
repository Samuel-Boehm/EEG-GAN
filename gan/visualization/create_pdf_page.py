# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

def add_pdf_page(pdf_pages, n_rows, n_cols, contents):
    assert len(contents) == n_rows * n_cols, "Mismatch in dimensions or content"

    # Create a new figure with A4 dimensions
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8.27, 11.69))

    # Wrap axs in a list if it's a single Axes object
    if n_rows == n_cols == 1:
        axs = [axs]

    # Flatten the axes array if necessary
    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()

    # Add content to the subplots
    for i, content in enumerate(contents):
        if isinstance(content, str):  # Text
            axs[i].text(0.5, 0.5, content, horizontalalignment='center', verticalalignment='center')
            axs[i].axis('off')  # Remove axes for text subplots
        elif isinstance(content, plt.Axes):  # Axes
            fig.add_axes(content)
        elif isinstance(content, pd.DataFrame):  # Table
            axs[i].axis('off')  # Remove axes for table subplots
            table = axs[i].table(cellText=content.values, colLabels=content.columns, cellLoc='center', loc='center')
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

    # Save the figure to the PdfPages object
    pdf_pages.savefig(fig, bbox_inches='tight')