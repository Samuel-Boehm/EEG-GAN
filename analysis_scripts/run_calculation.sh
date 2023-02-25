#!/bin/bash

stage=3

printf 'Running scripts for stage: %d\n' "$stage"


python3 calculate_metrics.py --name 'baseline' --data_name 'baseline' --stage $stage
python3 calculate_metrics.py --name 'ZCA' --data_name 'whitened' --stage $stage
python3 calculate_metrics.py --name 'cGAN' --data_name 'whitened' --stage $stage
python3 calculate_metrics.py --name 'cGAN_V2' --data_name 'whitened' --stage $stage
python3 calculate_metrics.py --name 'torch_cGAN' --data_name 'whitened' --stage $stage
python3 calculate_metrics.py --name 'spGAN' --data_name 'whitened' --stage $stage

echo 'All scripts ran'
