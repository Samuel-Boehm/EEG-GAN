# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import sys
import torch
import numpy as np
import os
from lightning import LightningModule
from torch.nn.functional import softplus
from torch import autograd
from gan.model.critic import Critic, build_critic
from gan.model.generator import Generator, build_generator
from gan.model.spectral_Critic import spectralCritic, build_sp_critic

class GAN(LightningModule):
    def __init__(self, n_channels, n_classes, n_time, n_stages, n_filters,
        fs, latent_dim:int = 100, lambda_gp:float = 10., lr_gen:float = 0.001,
        lr_critic:float = 0.005, b1:float = 0.0, b2:float = 0.999,
        batch_size:int = 32, epochs_per_stage:int = 200, 
        embedding_dim:int = 10, fading:bool = False, alpha:float = 1, beta:float = .2, **kwargs,
    ):  
        """
        Class for Generative Adversarial Network (GAN) for EEG data. This inherits from the
        LightningModule class and ist trained using the PyTorch Lightning Trainer framework.
        

        Args:
            n_channels (int):   number of EEG channels in the training data
            
            n_classes (int):    number of classes in the training data
            
            n_time (int):       number of time samples in the training data
            
            n_stages (int):     number of stages for the progressive growing of the GAN
            
            n_filters (int):    number of filters for the convolutional layers
            
            fs (int):           sampling frequency of the training data. 
                                This is just here so we can save it in the hparams dict.
            
            latent_dim (int, optional): dimension of the latent vector. Defaults to 100.
            
            lambda_gp (int, optional):lambda hyperparameter for the gradient penalty.
                Defaults to 10.
            
            lr_gen (float, optional): generator learning rate. Defaults to 0.001.
            
            lr_critic (float, optional): critic learning rate. Defaults to 0.005.
            
            b1 (float, optional): first decay rate for the Adam optimizer. Defaults to 0.0.
            
            b2 (float, optional): second decay rate for the Adam optimizer. Defaults to 0.999.
            
            batch_size (int, optional): batch size. Defaults to 32.
            
            epochs_per_stage (int or list, optional): number of epochs per stage. Total number of training
                stages is calculated by epochs_per:stage * n_stages. Defaults to 200 epochs per stage.
                If a list is passed, the length of the list must be equal to n_stages. With a list,
                the number of epochs per stage is determined by the values in the list.
            
            embedding_dim (int, optional): size of the embedding layer in the generator.
                Defaults to 10.

            fading (bool, optional): if True, fading between old and new blocks is used. Defaults to False.

            alpha (float, optional): alpha parameter to weight the amount of the time domain loss. Defaults to 1.

            beta (float, optional): beta parameter to weight the amount of the frequency domain loss. Defaults to .2.
        
        Methods:   
            forward(x): forward pass through the model
            training_step(batch, batch_idx): training step for the lightning module
            configure_optimizers(): configure the optimizers for the lightning module
            on_train_epoch_end(): method that is called at the end of each epoch
            gradient_penalty(critic, real, fake): calculate the gradient penalty

        For further methods LightningModule documentation.
        """
        
        super().__init__(**kwargs)

        # Save all hyperparameters
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.current_stage = 1

        # Build the generator model
        self.generator: Generator = build_generator(n_filters, n_time, n_stages, n_channels,
                                                    n_classes, latent_dim, embedding_dim, fading)
        
        # Build the critic model
        self.critic: Critic = build_critic(n_filters, n_time, n_stages, n_channels, n_classes, fading)

        # Build the spectral critic model
        self.sp_critic: spectralCritic = build_sp_critic(n_filters, n_time, n_stages, n_channels,
                                                         n_classes, fading,
                                                        )
        
        
        # Determine fading epochs
        # In these epochs a new block is added to the generator and the critic
        self.progression_epochs = []

        if isinstance(epochs_per_stage, int):
            for i in range(n_stages):
                self.progression_epochs.append(epochs_per_stage*i)
        elif isinstance(epochs_per_stage, list): 
            pass

            if len(epochs_per_stage) != n_stages:
                raise ValueError (
                f"len(epochs_per_stage) != n_stages. Number of stages must be equal to the"
                f"number of epochs per stage. Got {len(epochs_per_stage)} epochs and {n_stages} stages."
                f"{epochs_per_stage}")
            else:
                self.progression_epochs = np.cumsum(epochs_per_stage)
                self.progression_epochs = np.insert(self.progression_epochs, 0, 0)[:-1]
                # We need to add a 0 to the beginning of the list, because we need to trigger the 
                # 'set_stage' method at the beginning of the training.
                
        else:
            raise TypeError("epochs_per_stage must be either int or list")

        # For each metric we want to log, we need to initialize a list inside the metrics dict.
        
        self.generated_data = []
        self.real_data = []
        self.y_real = []
        self.y_fake = []
        self.loss_generator = []
        self.loss_critic = []
        self.gp = []
        self.fd_loss = []

        
    def forward(self, z, y):
        return self.generator(z, y)
        
    def training_step(self, batch_real, batch_idx):
        
        X_real, y_real = batch_real

        optimizer_g, optimizer_c, optimizer_spc = self.optimizers()

        # generate noise and labels:
        z = torch.randn(X_real.shape[0], self.hparams.latent_dim)
        z = z.type_as(X_real)

        # generate fake batch:
        y_fake = torch.randint(low=0, high=self.hparams.n_classes,
                               size=(X_real.shape[0],), dtype=torch.int32)
        
        y_fake  = y_fake.type_as(y_real)

        X_fake = self.forward(z, y_fake)

        # optimize critic in time domain
        c_loss, gp = self.train_critic(X_real, y_real, X_fake, y_fake, self.critic, optimizer_c)

        # optimize critic in frequency domain
        self.train_critic(X_real, y_real, X_fake, y_fake, self.sp_critic, optimizer_spc)

        ## optimize generator   
        fx_fake = self.critic(X_fake, y_fake)
        fx_spc = self.sp_critic(X_fake, y_fake)

        loss_td = softplus(-fx_fake).mean()
        loss_fd = softplus(-fx_spc).mean()

        g_loss = (self.hparams.alpha * loss_td + self.hparams.beta * loss_fd) / (self.hparams.alpha + self.hparams.beta)

        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        # Collect data during training for metrics:
        self.generated_data.append(X_fake.detach().cpu().numpy())
        self.real_data.append(X_real.detach().cpu().numpy())
        self.y_real.append(y_real.detach().cpu().numpy())
        self.y_fake.append(y_fake.detach().cpu().numpy())

        self.loss_generator.append(g_loss.item())
        self.loss_critic.append(c_loss.item())
        self.gp.append(gp.item())

        self.fd_loss.append(loss_fd.item())

        torch.cuda.empty_cache()


    def configure_optimizers(self):
        lr_gene = self.hparams.lr_gen
        lr_critic = self.hparams.lr_critic
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_gene, betas=(b1, b2))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, betas=(b1, b2))
        opt_spc = torch.optim.Adam(self.sp_critic.parameters(), lr=lr_critic, betas=(b1, b2))
        return [opt_g, opt_c, opt_spc], []
    

    def gradient_penalty(self, batch_real, batch_fake, y_fake, critic):
        """
		Improved WGAN gradient penalty

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

        critic_interpolates = critic(interpolates, y_fake)

        ones = torch.ones(critic_interpolates.size(), device=self.device)

        gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                    grad_outputs=ones,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        tmp = (gradients.norm(2, dim=1) - 1)

        gradient_penalty = ((tmp) ** 2).mean()

        return gradient_penalty
    
    def train_critic(self, X_real, y_real, X_fake, y_fake, critic, optim):

        fx_real = critic(X_real, y_real)
        fx_fake = critic(X_fake.detach(), y_fake.detach())
        gp = self.gradient_penalty(X_real, X_fake, y_fake, critic)

        c_loss = (
                -(torch.mean(fx_real) - torch.mean(fx_fake))
                + self.hparams.lambda_gp * gp 
                + (0.001 * torch.mean(fx_real ** 2))
           )
        
        optim.zero_grad()
        self.manual_backward(c_loss, retain_graph=True)
        optim.step()

        return c_loss, gp
    
    