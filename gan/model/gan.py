# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from typing import Tuple
import numpy as np
import os
from lightning import LightningModule
from typing import Tuple

import torch
from torch.nn.functional import softplus
from torch import autograd
import torch.nn as nn
from torch import Tensor

from src.metrics import SWD

from gan.model.critic import Critic
from gan.model.generator import Generator
from gan.model.spectral_critic import SpectralCritic

from dataclasses import dataclass, field, asdict

@dataclass
class OutputContainer:
    generated_data: list = field(init=False)
    real_data: list = field(init=False)
    y_real: list = field(init=False)
    y_fake: list = field(init=False)
    loss_generator: list = field(init=False)
    loss_critic: list = field(init=False)
    loss_sp_critic: list = field(init=False)
    gradient_penalty: list = field(init=False)

    def clear(self):
        for key, value in asdict(self).items():
            if isinstance(value, list):
                setattr(self, key, [])


class GAN(LightningModule):
    """
    Class for Generative Adversarial Network (GAN) for EEG data. This inherits from the
    LightningModule class and ist trained using the PyTorch Lightning Trainer framework.
    

    Parameters:
    ----------
        n_channels (int)
            number of EEG channels in the training data
        
        n_classes (int)
            number of classes in the training data
        
        n_time (int)
            number of time samples in the training data
        
        n_stages (int)
            number of stages for the progressive growing of the GAN
        
        n_filters (int)
            number of filters for the convolutional layers
        
        fs (int):
            sampling frequency of the training data. 
            This is just here so its collected in the hparams dict.
        
        latent_dim (int, optional)
            dimension of the latent vector. Defaults to 100.
        
        lambda_gp (int, optional)
            lambda hyperparameter for the gradient penalty. Defaults to 10.
        
        lr_generator (float, optional)
            generator learning rate. Defaults to 0.001.
        
        lr_critic (float, optional)
            critic learning rate. Defaults to 0.005.
        
        n_critic (int, optional)
            number of critic updates per generator update. Defaults to 5.
        
        b1 (float, optional)
            first decay rate for the Adam optimizer. Defaults to 0.0.
        
        b2 (float, optional)
            second decay rate for the Adam optimizer. Defaults to 0.999.
        
        batch_size (int, optional)
            batch size. Defaults to 32.
        
        epochs_per_stage (int or list, optional)
            number of epochs per stage. Total number of training stages is calculated by
            epochs_per:stage * n_stages. Defaults to 200 epochs per stage. In order to train stages
            with variable number of epochs, epochs_per_stage can be a list of integers. The length
            of the list must be equal to n_stages. 
        
        embedding_dim (int, optional)
            size of the embedding layer in the generator. Defaults to 10.

        fading (bool, optional)
            if True, fading between old and new blocks is used. Defaults to False.

        alpha (float, optional)
            alpha parameter to weight the amount of the time domain loss. Defaults to 1.

        beta (float, optional)
            beta parameter to weight the amount of the frequency domain loss. Defaults to .2.
    
    Methods:
    ----------
        forward(x)
            forward pass through the model

        training_step(batch, batch_idx)
            defines the training step for the lightning module

        configure_optimizers()
            configure the optimizers for the lightning module

        on_train_epoch_end()
            method that is called at the end of each epoch
        
        gradient_penalty(critic, real, fake)
            calculate the gradient penalty

    For further methods LightningModule documentation.
    """
    def __init__(self, 
                generator:Generator,
                critic:Critic,
                spectral_critic:SpectralCritic,
                optimizer: Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
                epochs_per_stage:int,
                n_stages:int,
                n_epochs_critic:int,
                alpha:float,
                beta:float,
                **kwargs
                 ):  
        
        super().__init__(**kwargs)
        self.generator = generator
        self.critic = critic
        self.sp_critic = spectral_critic
        self.optim = optimizer
        # Save all hyperparameters
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.cur_stage = 1
        
        self.oc = OutputContainer()
        
        
        # Determine transition epochs:
        self.progression_epochs = []

        if isinstance(epochs_per_stage, int):
            for i in range(n_stages):
                self.progression_epochs.append(epochs_per_stage*i)
        elif isinstance(epochs_per_stage, list): 
            if len(epochs_per_stage) != n_stages:
                raise ValueError (
                f"""
                len(epochs_per_stage) != n_stages. Number of stages must be equal to the
                number of epochs per stage. Got {len(epochs_per_stage)} epochs and {n_stages} stages.
                {epochs_per_stage}
                """)
            else:
                self.progression_epochs = np.cumsum(epochs_per_stage)
                self.progression_epochs = np.insert(self.progression_epochs, 0, 0)[:-1]
                # We need to add a 0 to the beginning of the list, in order to trigger the 
                # 'set_stage' method at the beginning of the training.     
        else:
            raise TypeError("epochs_per_stage must be either int or list of ints.")

        
    def forward(self, z, y):
        return self.generator(z, y)
        
    def training_step(self, batch_real:Tuple[Tensor, Tensor], batch_idx:int):
        """
        The training step for the GAN. This is called by the Lightning Trainer framework.
        Here we define a single training step for the GAN. This includes the forward pass,
        the calculation of the loss and the backward pass.

        Steps
        ----------
        1: Generate a batch of fake data
        2: In each epoch train critic
        3: In each n_critic epochs train generator

        """
        
        X_real, y_real = batch_real

        optim_g, optim_c, optim_spc = self.optimizers()
        
        # 1: Generate fake batch:
        X_fake, y_fake = self.generator.generate(X_real.shape)

        y_fake  = y_fake.type_as(y_real)
        
        # 2: Train critic:
        ## optimize time domain critic
        c_loss, gp = self.train_critic(X_real, y_real, X_fake, y_fake, self.critic, optim_c)

        ## optimize frequency domain critic
        spc_loss, _ =self.train_critic(X_real, y_real, X_fake, y_fake, self.sp_critic, optim_spc)

        # 3: Train generator:
        # If n_critic =! 1 we train the generator only every n_th step
        if batch_idx % self.hparams.n_epochs_critic == 0:
            ## optimize generator   
            fx_fake = self.critic(X_fake, y_fake)
            fx_spc = self.sp_critic(X_fake, y_fake)

            loss_td = softplus(-fx_fake).mean()
            loss_fd = softplus(-fx_spc).mean()

            g_loss = (self.hparams.alpha * loss_td + self.hparams.beta * loss_fd) / (self.hparams.alpha + self.hparams.beta)

            optim_g.zero_grad()
            self.manual_backward(g_loss)
            optim_g.step()

            self.oc.loss_generator.append(g_loss.item())

        # Log
        self.log()

        # Collect data during training for metrics:
        self.oc.generated_data.append(X_fake.detach().cpu().numpy())
        self.oc.real_data.append(X_real.detach().cpu().numpy())
        self.oc.y_real.append(y_real.detach().cpu().numpy())
        self.oc.y_fake.append(y_fake.detach().cpu().numpy())
        self.oc.gradient_penalty.append(gp.detach().cpu().numpy())
        self.oc.loss_critic.append(c_loss.detach().cpu().numpy())
        self.oc.loss_sp_critic.append(spc_loss.detach().cpu().numpy())


    def configure_optimizers(self):
        lr_generator = self.hparams.lr_gen
        lr_critic = self.hparams.lr_critic
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        if self.hparams.optimizer == 'adam':
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_generator, betas=(b1, b2))
            opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, betas=(b1, b2))
            opt_spc = torch.optim.Adam(self.sp_critic.parameters(), lr=lr_critic, betas=(b1, b2))
        else:
            raise ValueError(f"Optimizer {self.hparams.optimizer} not supported")
        return [opt_g, opt_c, opt_spc], []
    

    def gradient_penalty(self, batch_real:Tuple[Tensor, Tensor],
                         batch_fake:Tuple[Tensor, Tensor], y_fake, critic) -> Tensor:
        """
		Improved WGAN gradient penalty from Hartmann et al. (2018)
        https://arxiv.org/abs/1806.01875

        Gradient pentaly is calculated by interpolating between real and fake data. The interpolated
        data is then passed to the critic and the gradient of the critic's output with respect to 
        the interpolated data is calculated. The gradient penalty is then the squared norm of the
        gradient minus one. 

		Parameters
		----------
		batch_real : tensor
			Batch of real data
		batch_fake : tensor
			Batch of fake data
        y_fake : tensor
            Labels for the fake data - just to pass along to the critic
        critic : torch.nn.Module
            current critic model to calculate penalty for


		Returns
		-------
		gradient_penalty : tensor
			Gradient penalties
		"""

        alpha = torch.rand(batch_real.size(0),*((len(batch_real.size())-1)*[1]), device=self.device)
        alpha = alpha.expand(batch_real.size())

        interpolates = alpha * batch_real + ((1 - alpha) * batch_fake)

        output_interpolates = critic(interpolates, y_fake)

        ones = torch.ones(output_interpolates.size(), device=self.device)

        gradients = autograd.grad(outputs=output_interpolates, inputs=interpolates,
                                    grad_outputs=ones,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    
    def train_critic(self, X_real:Tensor, y_real:Tensor,
                     X_fake:Tensor, y_fake:Tensor, critic:nn.Module, optim:torch.optim.Optimizer) -> Tuple[Tensor, Tensor]:
        """
        Trains the critic (discriminator) part of the GAN.

        Parameters
        ----------
        X_real (Tensor): Real samples.
        y_real (Tensor): Labels for the real samples.
        X_fake (Tensor): Generated (fake) samples.
        y_fake (Tensor): Labels for the fake samples.
        critic (nn.Module): The critic model.
        optim (Optimizer): The optimizer for the critic.

        Returns
        ----------
        c_loss (Tensor): The critic loss.
        gp (Tensor): The gradient penalty.

        The function calculates the critic's output for both real and fake samples, 
        computes the gradient penalty and the critic loss, and performs a backward pass and an optimization step.
        """

        fx_real = critic(X_real, y_real)
        fx_fake = critic(X_fake.detach(), y_fake.detach())
       

        distance = torch.mean(fx_fake) - torch.mean(fx_real)

        # Relaxed gradient penalty following Hartmann et al. (2018)
        # https://arxiv.org/abs/1806.01875
        # gradient pentalty is scaled with the distance between the distrubutions and a lambda hyperparameter
        # Note: Hartmann et al suggest to scale the gradient penalty with the distance between the distributions
        # this does not work well so far... (?)
        gp = self.hparams.lambda_gp * self.gradient_penalty(X_real, X_fake, y_fake, critic)

        c_loss = distance + gp        
        
        optim.zero_grad()
        self.manual_backward(c_loss, retain_graph=True)
        optim.step()

        return c_loss, gp