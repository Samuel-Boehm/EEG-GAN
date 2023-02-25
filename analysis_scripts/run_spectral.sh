#!/bin/bash

stage=4

printf 'Running scripts for stage: %d\n' "$stage"

python3 calculate_spectral_statistics.py --name 'baseline' --data_name 'baseline' --stage $stage
python3 calculate_spectral_statistics.py --name 'ZCA' --data_name 'whitened' --stage $stage
python3 calculate_spectral_statistics.py --name 'cGAN' --data_name 'whitened' --stage $stage
python3 calculate_spectral_statistics.py --name 'cGAN_V2' --data_name 'whitened' --stage $stage
python3 calculate_spectral_statistics.py --name 'torch_cGAN' --data_name 'whitened' --stage $stage
python3 calculate_spectral_statistics.py --name 'spGAN' --data_name 'whitened' --stage $stage

printf 'All scripts ran for stage  %d\n' "$stage"
