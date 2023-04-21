#  Author: Samuel Böhm <samuel-boehm@web.de>
import torch
from torch.nn.functional import softplus
from torch import autograd

from eeggan.cuda import to_device
from eeggan.data.dataclasses import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.trainer.utils import detach_all
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from torch.optim.optimizer import Optimizer


def gradient_penalty(discriminator, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], device):

    batch_size = batch_real.X.shape[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(batch_real.X).to(device)
    # getting x hat
    interpolated = alpha * batch_real.X + (1 - alpha) * batch_fake.X

    dis_interpolated = discriminator(interpolated, batch_fake.y)
    grad_outputs = torch.ones(dis_interpolated.shape).to(device)

    # Calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=dis_interpolated, inputs=interpolated,
                           grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, n_samples),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = ((torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12) - 1) ** 2).mean()
    return gradients_norm


class SpectralTrainer(GanSoftplusTrainer):
    
    def __init__(self, i_logging: int, discriminator: Discriminator, generator: Generator, r1: float,
                 r2: float, spectral_discriminator: Discriminator):
                 
        self.spectral_discriminator = spectral_discriminator
        super().__init__(i_logging, discriminator, generator, r1, r2)


    def _train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor],
                            discriminator, optimizer):
        
        batch_real.X.requires_grad_(True)
        batch_fake.X.requires_grad_(True)

        discriminator_real = discriminator(batch_real.X, batch_real.y).reshape(-1)
        discriminator_fake = discriminator(batch_fake.X, batch_fake.y).reshape(-1)
        gp = gradient_penalty(discriminator, batch_real, batch_fake, device=batch_real.X.device)
        loss_discriminator = (-(torch.mean(discriminator_real) - torch.mean(discriminator_fake)) + self.r1_gamma * gp)
        
        discriminator.zero_grad()
        loss_discriminator.backward(retain_graph=True)
        optimizer.step()

        return loss_discriminator.item(),  gp


    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        # Train time domain discriminator
        loss_td, gp_td = self._train_discriminator(batch_real, batch_fake, self.discriminator, self.optim_discriminator)
        # Train the spectral discriminator
        loss_fd, gp_td = self._train_discriminator(batch_real, batch_fake, self.spectral_discriminator, self.optim_spectral)

        return {'loss_td': loss_td, 'gp_td': gp_td,
                'loss_fd': loss_fd, 'gp_fd':gp_td}

    def train_generator(self, batch_real: Data[torch.Tensor]):
        self.generator.zero_grad()
        self.optim_generator.zero_grad()
        self.generator.train(True)
        self.discriminator.train(False)

        with torch.no_grad():
            latent, y_fake, y_onehot_fake = to_device(batch_real.X.device,
                                                      *self.generator.create_latent_input(self.rng, len(batch_real.X)))
            latent, y_fake, y_onehot_fake = detach_all(latent, y_fake, y_onehot_fake)

        X_fake = self.generator(latent.requires_grad_(False), y_fake)
        batch_fake = Data[torch.Tensor](X_fake, y_fake, y_onehot_fake)


        fx_fake = self.discriminator(batch_fake.X.requires_grad_(True), batch_fake.y)
        
        sp_fake = self.spectral_discriminator(batch_fake.X.requires_grad_(True), batch_fake.y)
        
        
        # loss_td = softplus(-fx_fake).mean()
        # loss_fd = softplus(-sp_fake).mean()
        
        a = 1
        b = .2
        
        # err = (a*fx_fake + b*sp_fake) / (a+b)
        loss_ = softplus(-sp_fake).mean()
        loss = softplus(-fx_fake).mean()
        
        loss_c = (a*loss + b*loss_) / (a+b)
    
        # optimize
          

        loss_c.backward()

        self.optim_generator.step()

        return {'td':loss.item(), 'fd':loss_.item(), 'combined': loss_c.item()}

    def set_optimizers(self, optim_discriminator: Optimizer, optim_generator: Optimizer, optim_spectral: Optimizer = None):
        self.optim_discriminator = optim_discriminator
        self.optim_generator = optim_generator
        self.optim_spectral = optim_spectral


