# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from typing import Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor, autograd
from torch.nn.functional import softplus
from torchmetrics import MeanMetric

from src.metrics import SWD
from src.models.components import Critic, Generator, SpectralCritic


class GAN(LightningModule):
    """
    Class for Generative Adversarial Network (GAN) for EEG data. This inherits from the
    LightningModule class and ist trained using the PyTorch Lightning Trainer framework.


    Parameters:
    ----------
        generator (Generator)
            Generator model for the GAN

        critic (Critic)
            Critic model for the GAN

        spectral_critic (SpectralCritic) (optional)
            Spectral critic model for the GAN

        n_epochs_critic (int)
            number of epochs to train the critic

        alpha (float, optional)
            alpha parameter to weight the amount of the time domain loss.

        beta (float, optional)
            beta parameter to weight the amount of the frequency domain loss.

        lambda_gp (float, optional)
            lambda parameter for the gradient penalty.

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
    """

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        n_epochs_critic: int,
        alpha: float,
        beta: float,
        lambda_gp: float,
        optimizer: DictConfig,
        spectral_critic: Optional[SpectralCritic] = None,
        softplus: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.critic = critic
        self.sp_critic = spectral_critic
        self.optimizer_dict = optimizer
        self.automatic_optimization = False
        self.current_stage = 1
        self.lambda_gp = lambda_gp
        self.n_epochs_critics = n_epochs_critic
        self.alpha = alpha
        self.beta = beta
        self.softplus = softplus

        self.sliced_wasserstein_distance = SWD()
        self.generator_loss = MeanMetric()
        self.critic_loss = MeanMetric()
        if self.sp_critic:
            self.sp_critic_loss = MeanMetric()
        self.gp = MeanMetric()
        # self.generator_alpha = MeanMetric()
        # self.critic_alpha = MeanMetric()

    def forward(self, z, y):
        return self.generator(z, y)

    def training_step(self, batch_real: Tuple[Tensor, Tensor], batch_idx: int):
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
        X_fake, y_fake = self.generator.generate(X_real.shape[0])
        y_fake = y_fake.type_as(y_real)

        # 2: Train critic:
        c_loss, gp = self.train_critic(
            X_real, y_real, X_fake, y_fake, self.critic, optim_c
        )

        spc_loss = None
        if self.sp_critic:
            spc_loss, _ = self.train_critic(
                X_real, y_real, X_fake, y_fake, self.sp_critic, optim_spc
            )

        # 3: Train generator (every n_critic steps)
        if (batch_idx + 1) % self.n_epochs_critics == 0:
            self.toggle_optimizer(optim_g)

            # Reuse the same fake batch for generator update
            fx_fake = self.critic(X_fake, y_fake)
            if self.sp_critic:
                fx_spc = self.sp_critic(X_fake, y_fake)
                loss_fd = torch.mean(softplus(-fx_spc))
                beta = self.beta
            else:
                loss_fd = 0
                beta = 0

            loss_td = torch.mean(softplus(-fx_fake))

            # Avoid division by zero if both alpha and beta are zero
            alpha_beta_sum = self.alpha + beta
            if alpha_beta_sum == 0:
                raise ValueError("Sum of alpha and beta for generator loss is zero!")
            g_loss = (self.alpha * loss_td + beta * loss_fd) / alpha_beta_sum

            self.manual_backward(g_loss)
            optim_g.step()
            optim_g.zero_grad()
            self.untoggle_optimizer(optim_g)
            self.generator_loss(g_loss)

        # Log
        self.critic_loss(c_loss.item())
        if self.sp_critic and spc_loss is not None:
            self.sp_critic_loss(spc_loss.item())
        self.gp(gp.item())

        self.sliced_wasserstein_distance.update(X_real, X_fake)

        self.log_metrics()

    def configure_optimizers(self):
        lr_generator = self.optimizer_dict.lr_generator
        lr_critic = self.optimizer_dict.lr_critic
        betas = self.optimizer_dict.betas

        if self.optimizer_dict.name == "adam":
            opt_g = torch.optim.Adam(
                self.generator.parameters(), lr=lr_generator, betas=(betas)
            )
            opt_c = torch.optim.Adam(
                self.critic.parameters(), lr=lr_critic, betas=(betas)
            )
            if self.sp_critic:
                opt_spc = torch.optim.Adam(
                    self.sp_critic.parameters(), lr=lr_critic, betas=(betas)
                )
            else:
                opt_spc = None
        else:
            raise ValueError(f"Optimizer {self.optimizer_dict.name} not supported")

        return [opt_g, opt_c, opt_spc], []

    def gradient_penalty(
        self, real: Tensor, fake: Tensor, y_fake, critic: Critic
    ) -> Tensor:
        """

                Improved WGAN gradient penalty from Hartmann et al. (2018)
        https://arxiv.org/abs/1806.01875

        Gradient pentaly is calculated by interpolating between real and fake data. The interpolated
        data is then passed to the critic and the gradient of the critic's output with respect to
        the interpolated data is calculated. The gradient penalty is then the squared norm of the
        gradient minus one.

                Parameters
                ----------
                real : tensor
                        Batch of real data
                fake : tensor
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

        alpha = torch.rand(
            real.size(0), *((len(real.size()) - 1) * [1]), device=self.device
        )
        alpha = alpha.expand(real.size())

        interpolates = alpha * real + ((1 - alpha) * fake).requires_grad_(True)

        output_interpolates = critic(interpolates, y_fake)

        ones = torch.ones(
            output_interpolates.size(), device=self.device, requires_grad=False
        )

        gradients = autograd.grad(
            outputs=output_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

        # just in case:
        del gradients, interpolates, output_interpolates, alpha, ones

        return gradient_penalty

    def train_critic(
        self,
        X_real: Tensor,
        y_real: Tensor,
        X_fake: Tensor,
        y_fake: Tensor,
        critic: nn.Module,
        optim: torch.optim.Optimizer,
    ) -> Tuple[Tensor, Tensor]:
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
        self.toggle_optimizer(optim)
        fx_real = critic(X_real, y_real)
        fx_fake = critic(X_fake.detach(), y_fake)

        # Relaxed gradient penalty following Hartmann et al. (2018)
        # https://arxiv.org/abs/1806.01875
        # gradient pentalty is scaled with the distance between the distrubutions and a lambda hyperparameter
        # Note: Hartmann et al suggest to scale the gradient penalty with the distance between the distributions
        # this does not work well so far... (?)
        gp = self.lambda_gp * self.gradient_penalty(X_real, X_fake, y_fake, critic)

        c_loss = torch.mean(softplus(-fx_real)) + torch.mean(softplus(fx_fake)) + gp

        self.manual_backward(c_loss, retain_graph=True)
        optim.step()
        optim.zero_grad()
        self.untoggle_optimizer(optim)
        return c_loss, gp

    def on_train_start(self) -> None:
        self.reset_metrics()

    def log_metrics(self) -> None:
        self.log(
            "generator loss",
            self.generator_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "critic loss",
            self.critic_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        if self.sp_critic:
            self.log(
                "spectral critic loss",
                self.sp_critic_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
        self.log(
            "gradient penalty", self.gp, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log(
            "slice wasserstein distance",
            self.sliced_wasserstein_distance,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

    def reset_metrics(self) -> None:
        self.generator_loss.reset()
        self.critic_loss.reset()
        self.sp_critic_loss.reset()
        self.gp.reset()
        self.sliced_wasserstein_distance.reset()
