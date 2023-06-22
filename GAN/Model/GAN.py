# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch
import os
from lightning import LightningModule
from torch.nn.functional import softplus
from torch import autograd
from Model.Critic import Critic, build_critic
from Model.Generator import Generator, build_generator
from Handler.VisualizationHandler import VisualizationHandler

class GAN(LightningModule, VisualizationHandler):
    def __init__(self, n_channels, n_classes, n_time, n_stages, n_filters,
        fs, latent_dim: int = 100, lambda_gp = 10, lr_gen: float = 0.001,
        lr_critic: float = 0.005, b1: float = 0.0, b2: float = 0.999,
        batch_size: int = 32, epochs_per_stage: int = 200, plot_interval: int = 200,
        embedding_dim: int = 10, **kwargs,
    ):  
        """Class for Generative Adversarial Network (GAN) for EEG data. This inherits from the
        LightningModule class and ist trained using the PyTorch Lightning Trainer framework.
        Further it inherits from the VisualizationHandler class to provide visualization methods.

        Args:
            n_channels (int): number of EEG channels in the training data
            
            n_classes (int): number of classes in the training data
            
            n_time (int): number of time samples in the training data
            
            n_stages (int): number of stages for the progressive growing of the GAN
            
            n_filters (int): number of filters for the convolutional layers
            
            fs (int): sampling frequency of the training data. Needed to calculate the frequency 
                during each stage.
            
            latent_dim (int, optional): dimension of the latent vector. Defaults to 100.
            
            lambda_gp (int, optional): lambda hyperparameter for the gradient penalty.
                Defaults to 10.
            
            lr_gen (float, optional): generator learning rate. Defaults to 0.001.
            
            lr_critic (float, optional): critic learning rate. Defaults to 0.005.
            
            b1 (float, optional): first decay rate for the Adam optimizer. Defaults to 0.0.
            
            b2 (float, optional): second decay rate for the Adam optimizer. Defaults to 0.999.
            
            batch_size (int, optional): batch size. Defaults to 32.
            
            epochs_per_stage (int, optional): number of epochs per stage. Total number of training
                stages is calculated by epochs_per:stage * n_stages. Defaults to 200.
            
            plot_interval (int, optional): interval for plotting. Defaults to 200.
            
            embedding_dim (int, optional): size of the embedding layer in the generator.
                Defaults to 10.
        
        Methods:   
            forward(x): forward pass through the model
            training_step(batch, batch_idx): training step for the lightning module
            configure_optimizers(): configure the optimizers for the lightning module
            on_train_epoch_end(): method that is called at the end of each epoch
            gradient_penalty(critic, real, fake): calculate the gradient penalty

        For further methods see the VisualizationHandler and LightningModule classes.
        """
        
        super().__init__(**kwargs)

        # Save all hyperparameters
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.current_stage = 1

        # Build the generator model
        self.generator: Generator = build_generator(n_filters, n_time, n_stages, n_channels,
                                                    n_classes, latent_dim, embedding_dim,)
        
        # Build the critic model
        self.critic: Critic = build_critic(n_filters, n_time, n_stages, n_channels, n_classes,)
        
        
        # Determine fading epochs
        # In these epochs a new block is added to the generator and the critic
        self.progression_epochs = []
        for i in range(n_stages):
            self.progression_epochs.append(int(i*epochs_per_stage))
        
        # Init lists for real and fake data, this is used for plotting

        self.generated_data = []
        self.real_data = []

        
    def forward(self, z, y):
        return self.generator(z, y)
        
    def training_step(self, batch_real, batch_idx):
        
        X_real, y_real = batch_real

        optimizer_g, optimizer_c = self.optimizers()

        # generate noise and labels:
        z = torch.randn(X_real.shape[0], self.hparams.latent_dim)
        z = z.type_as(X_real)

        # generate fake batch:
        y_fake = torch.randint(low=0, high=self.hparams.n_classes,
                               size=(X_real.shape[0],), dtype=torch.int32)
        
        y_fake  = y_fake.type_as(y_real)

        X_fake = self.forward(z, y_fake)

        # train critic
        self.toggle_optimizer(optimizer_c)
        
        fx_real = self.critic(X_real, y_real)
        fx_fake = self.critic(X_fake.detach(), y_fake)
        gp = self.gradient_penalty(X_real, X_fake, y_fake)

        loss_critic = (
                -(torch.mean(fx_real) - torch.mean(fx_fake))
                + self.hparams.lambda_gp * gp
                + (0.001 * torch.mean(fx_real ** 2))
            )

        self.log("critic_loss", loss_critic, prog_bar=True)
        self.manual_backward(loss_critic, retain_graph=True)

        optimizer_c.step()
        optimizer_c.zero_grad()
        self.untoggle_optimizer(optimizer_c)

        # train generator
        self.toggle_optimizer(optimizer_g)
        
        fx_fake = self.critic(X_fake, y_fake)
        g_loss = softplus(-fx_fake).mean()
        self.log("generator_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.generated_data.append(X_fake)
        self.real_data.append(X_real)
    
    def on_train_epoch_end(self):
        batch_real = torch.cat(self.real_data)
        batch_fake = torch.cat(self.generated_data)

        # each 'plot_intervall' stages plot the spectrum of the generated and real data
        if self.trainer.current_epoch % self.hparams.plot_interval == 0:
            # Set plotting params:
            max_freq = int(self.hparams.fs / 2**(self.hparams.n_stages - self.current_stage))
            plot_dir = os.path.join(self.logger.log_dir, "plots")

            self.plot_spectrum(batch_real, batch_fake, self.trainer.current_epoch,
                                max_freq, plot_dir,)
     
        self.real_data.clear()
        self.generated_data.clear()

    def configure_optimizers(self):
        lr_gene = self.hparams.lr_gen
        lr_critic = self.hparams.lr_critic
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_gene, betas=(b1, b2))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, betas=(b1, b2))
        return [opt_g, opt_c], []
    

    def gradient_penalty(self, batch_real, batch_fake, y_fake):
        """
		Improved WGAN gradient penalty

		Parameters
		----------
		batch_real : tensor
			Batch of real data
		batch_fake : tensor
			Batch of fake data

		Returns
		-------
		gradient_penalty : tensor
			Gradient penalties
		"""

        alpha = torch.rand(batch_real.size(0),*((len(batch_real.size())-1)*[1]), device=self.device)
        alpha = alpha.expand(batch_real.size())

        interpolates = alpha * batch_real + ((1 - alpha) * batch_fake)

        critic_interpolates = self.critic(interpolates, y_fake)

        ones = torch.ones(critic_interpolates.size(), device=self.device)

        gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                    grad_outputs=ones,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        tmp = (gradients.norm(2, dim=1) - 1)

        gradient_penalty = ((tmp) ** 2).mean()

        return gradient_penalty
