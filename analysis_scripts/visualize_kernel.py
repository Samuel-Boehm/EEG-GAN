# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from gan.model.gan import GAN
import wandb
import os 
import torch
from gan.paths import results_path, data_path
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np

path = "EEGGAN/kabqciz0"
api = wandb.Api()

run = api.run(path)
config = run.config

model_torch = GAN(**config)

checkpoint_path=os.path.join(results_path, path, "checkpoints/last.ckpt")

model_wb = GAN.load_from_checkpoint(
                checkpoint_path=os.path.join(results_path, path, "checkpoints/last.ckpt"),
                map_location=torch.device('cpu'),
                )

#model_torch.generator.load_state_dict(torch.load("/home/boehms/eeg-gan/EEG-GAN/temp_plots/generator.pth", 
#                                        map_location='cpu'),
#                            )

model_torch.eval()
model_wb.eval()


if __name__ == "__main__":
    f1_cp = (model_wb.generator.blocks[0].intermediate_sequence[4].conv2.weight.data.clone().flatten().detach().numpy())
    f2_cp = (model_wb.generator.blocks[1].intermediate_sequence[1].conv2.weight.data.clone().flatten().detach().numpy())
    f3_cp = (model_wb.generator.blocks[2].intermediate_sequence[1].conv2.weight.data.clone().flatten().detach().numpy())

    f1_csv = (np.loadtxt("/home/boehms/eeg-gan/EEG-GAN/temp_plots/block1_conv2.csv"))
    f2_csv = (np.loadtxt("/home/boehms/eeg-gan/EEG-GAN/temp_plots/block2_conv2.csv"))
    f3_csv = (np.loadtxt("/home/boehms/eeg-gan/EEG-GAN/temp_plots/block3_conv2.csv"))

    f1_torch = (model_torch.generator.blocks[0].intermediate_sequence[4].conv2.weight.data.clone().flatten().detach().numpy())
    f2_torch = (model_torch.generator.blocks[1].intermediate_sequence[1].conv2.weight.data.clone().flatten().detach().numpy())
    f3_torch = (model_torch.generator.blocks[2].intermediate_sequence[1].conv2.weight.data.clone().flatten().detach().numpy())

    
    weights_list = [f1_cp, f2_cp, f3_cp, f1_csv, f2_csv, f3_csv, f1_torch, f2_torch, f3_torch]


    plt.plot(f1_cp, label="w and b ", alpha=0.2)
    plt.plot(f1_csv, label="csv", alpha=0.2)
    plt.plot(f1_torch, label='torch', alpha =0.2)
    plt.legend()
    plt.savefig("f1.png")
    plt.clf()

    plt.plot(f2_cp, label="w and b ", alpha=0.2)
    plt.plot(f2_csv, label="csv", alpha=0.2)
    plt.plot(f2_torch, label='torch', alpha =0.2)
    plt.legend()
    plt.savefig("f2.png")
    plt.clf()


    plt.plot(f3_cp, label="w and b ", alpha=0.2)
    plt.plot(f3_csv, label="csv", alpha=0.2)
    plt.plot(f3_torch, label='torch', alpha =0.2)
    plt.legend()
    plt.savefig("f3.png")
    plt.clf()