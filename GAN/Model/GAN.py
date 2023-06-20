# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

import torch
from lightning import LightningModule
from torch.nn.functional import softplus
from torch import autograd
from Model.Critic import Critic, build_critic
from Model.Generator import Generator, build_generator
from Handler.VisualizationHandler import VisualizationHandler

class GAN(LightningModule, VisualizationHandler):
    def __init__(
        self,
        n_channels,
        n_classes,
        n_time,
        n_stages,
        n_filters,
        fs,
        plot_path:str,
        latent_dim: int = 100,
        lambda_gp = 10,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 32,
        epochs_per_stage: int = 200,
        plot_interval: int = 5,
        **kwargs,
    ):  
        VisualizationHandler.__init__(self, filepath=plot_path, fs=fs)
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.current_stage = 1
        self.embedding_dim = 10

        # networks
        self.generator: Generator = build_generator(
                                        latent_dim=latent_dim,
                                        embedding_dim=self.embedding_dim,
                                        n_filters=n_filters,
                                        n_time=n_time,
                                        n_stages=n_stages,
                                        n_channels=n_channels,
                                        n_classes=n_classes,
                                        )
        
        self.critic: Critic = build_critic(n_filters=n_filters,
                                         n_time=n_time,
                                         n_stages=n_stages,
                                         n_channels=n_channels,
                                         n_classes=n_classes,
                                         )
        
        
        # in these epochs a new block is added
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
        y_fake = torch.randint(low=0, high=self.hparams.n_classes, size=(X_real.shape[0],), dtype=torch.int32)
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
        if self.trainer.current_epoch % self.hparams.epochs_per_stage == 0:
            self.plot_spectrum(batch_real, batch_fake, self.trainer.current_epoch)
     
        self.real_data.clear()
        self.generated_data.clear()

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
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
