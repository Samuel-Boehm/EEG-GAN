# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from legacy_code.metrics.IntermediateOutput import IntermediateOutputWrapper
from legacy_code.data.batch import batch_data
from legacy_code.metrics.correlation import calculate_correlation_for_condition
from legacy_code.visualization.time_domain_plot import plot_time_domain
from legacy_code.visualization.stft_plots import plot_bin_stats
from legacy_code.visualization.spectrum_plots import plot_spectrum
from legacy_code.paths import data_path, results_path
from legacy_code.utils import generate_data

from braindecode.models.deep4 import Deep4Net
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import torch
import os
from legacy_code.visualization.create_pdf_page import add_pdf_page

parser = argparse.ArgumentParser()

parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-stage', '--stage', type=int, required=True)
parser.add_argument('-nsamp', '--n_samples', type=int, required=True)
parser.add_argument('-rn', '--report_name', type=str, required=True)
parser.add_argument('-subj', '--subject', type=int, required=False, default=0)

args = parser.parse_args()

mapping = {'right': 0, 'rest': 1}

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']

real = torch.load(os.path.join(data_path, 'clinical'))

if args.subject != 0:
    real.select_from_tag('subject', args.subject)
    # If a subset is selected, we generate data with the same number of samples:
    args.n_samples = real.data.shape[0]

# Subsample the real data.
idx = np.random.choice(real.data.shape[0], args.n_samples, replace=False)
real.data = real.data[idx]
real.target = real.target[idx]


# Generate fake data:
fake = generate_data(args.model_path, args.stage, args.n_samples)

# Metrics work best with batched data:
batch = batch_data(real.data, fake.data, real.target, fake.target)

# Create a PdfPages object
pdf_pages = PdfPages('model_report.pdf')

# Add a header to the page:
header = f'This is the report to {args.report_name}'


# Convert DataFrame to matplotlib figure and save to the PdfPages object
correlations = calculate_correlation_for_condition(batch, mapping)

add_pdf_page(pdf_pages, 2, 1, [header, correlations])

# Plot spectrum:
spectrum, _ = plot_spectrum(batch.real, batch.fake, fake.fs, 'all')
pdf_pages.savefig(spectrum, bbox_inches='tight')

# Plot time domain
TD_plot, _    = plot_time_domain(batch, channels, fake.fs, None)
pdf_pages.savefig(TD_plot, bbox_inches='tight')

for key in mapping.keys():
    conditional_real = batch.real[batch.y_real == mapping[key]]
    conditional_fake = batch.fake[batch.y_fake == mapping[key]]

    # calculate frequency in current stage:   
    fig_stats, fig_real, fig_fake = plot_bin_stats(conditional_real, conditional_fake,
                    fake.fs, channels, None, str(key), False)
    pdf_pages.savefig(fig_stats, bbox_inches='tight')
    pdf_pages.savefig(fig_real, bbox_inches='tight')    
    pdf_pages.savefig(fig_fake, bbox_inches='tight')

# Close the PdfPages object
pdf_pages.close()