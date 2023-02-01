#  Author: Samuel Böhm <samuel-boehm@web.de>
import torch
from torch.nn.functional import softplus
from torch import autograd

from eeggan.cuda import to_device
from eeggan.data.dataset import Data
from eeggan.training.discriminator import Discriminator
from eeggan.training.generator import Generator
from eeggan.training.trainer.utils import detach_all
from eeggan.training.trainer.gan_softplus import GanSoftplusTrainer
from torch.optim.optimizer import Optimizer


def gradient_penalty(D, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], device):

    batch_size = batch_real.X.shape[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(batch_real.X).to(device)
    # getting x hat
    interpolated = alpha * batch_real.X + (1 - alpha) * batch_fake.X

    dis_interpolated = D(interpolated, y=batch_real.y, y_onehot=batch_real.y_onehot)
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

def calc_gradient_penalty(X: torch.Tensor, y: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    inputs = X
    ones = torch.ones_like(outputs)
    if y is not None:
        inputs = (inputs, y)
        outputs = (outputs, outputs)
        ones = (ones, ones)
    gradients = autograd.grad(outputs=outputs, inputs=inputs,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)
    gradients = torch.cat([tmp.reshape(tmp.size(0), -1) for tmp in gradients], 1)
    gradient_penalty = 0.5 * gradients.norm(2, dim=1).pow(2).mean()
    return gradient_penalty

change_penalty = 0

class SpectralTrainer(GanSoftplusTrainer):
    
    def __init__(self, i_logging: int, discriminator: Discriminator, generator: Generator, r1: float,
                 r2: float, spectral_discriminator: Discriminator):
                 
        self.spectral_discriminator = spectral_discriminator
        super().__init__(i_logging, discriminator, generator, r1, r2)


    def _train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor],
                            discriminator, optimizer):
        discriminator.zero_grad()
        optimizer.zero_grad()
        discriminator.train(True)

        has_r1 = self.r1_gamma > 0.
        fx_real = discriminator(batch_real.X.requires_grad_(has_r1), y=batch_real.y.requires_grad_(has_r1),
                                     y_onehot=batch_real.y_onehot.requires_grad_(has_r1))

        loss_real = softplus(-fx_real).mean()
        loss_real.backward(retain_graph=has_r1)
        loss_r1 = None

        if has_r1:
            r1_penalty = self.r1_gamma * calc_gradient_penalty(batch_real.X.requires_grad_(True),
                                                               batch_real.y_onehot.requires_grad_(True), fx_real)
            r1_penalty.backward()
            loss_r1 = r1_penalty.item()

        has_r2 = self.r2_gamma > 0.
        
        fx_fake = discriminator(batch_fake.X.requires_grad_(has_r2), y=batch_fake.y.requires_grad_(has_r2),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(has_r2))
        loss_fake = softplus(fx_fake).mean()
        loss_fake.backward(retain_graph=has_r2)
        loss_r2 = None
        if has_r2:
            r2_penalty = self.r1_gamma * calc_gradient_penalty(batch_fake.X.requires_grad_(True),
                                                               batch_fake.y_onehot.requires_grad_(True), fx_real)
            r2_penalty.backward()
            loss_r2 = r2_penalty.item()

        optimizer.step()

        return loss_real.item(), loss_fake.item(), loss_r1, loss_r2


    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        # Train time domain discriminator
        loss_real_td, loss_fake_td, loss_r1_td, loss_r2_td = self._train_discriminator(batch_real, batch_fake, self.discriminator, self.optim_discriminator)
        # Train the spectral discriminator
        loss_real_fd, loss_fake_fd, loss_r1_fd, loss_r2_fd = self._train_discriminator(batch_real, batch_fake, self.spectral_discriminator, self.optim_spectral)



        return {'loss_real_td': loss_real_td, 'loss_fake_td': loss_fake_td, 'loss_r1_td': loss_r1_td, 'loss_r2_td':loss_r2_td,
                'loss real_fd': loss_real_fd, 'loss_fake_fd': loss_fake_fd, 'loss_r1_fd': loss_r1_fd, 'loss_r2_fd': loss_r2_fd,
                }

    def train_generator(self, batch_real: Data[torch.Tensor]):
        self.generator.zero_grad()
        self.optim_generator.zero_grad()
        self.generator.train(True)
        self.discriminator.train(False)

        with torch.no_grad():
            latent, y_fake, y_onehot_fake = to_device(batch_real.X.device,
                                                      *self.generator.create_latent_input(self.rng, len(batch_real.X)))
            latent, y_fake, y_onehot_fake = detach_all(latent, y_fake, y_onehot_fake)

        X_fake = self.generator(latent.requires_grad_(False), y=y_fake.requires_grad_(False),
                                y_onehot=y_onehot_fake.requires_grad_(False))
        batch_fake = Data[torch.Tensor](X_fake, y_fake, y_onehot_fake)


        fx_fake = self.discriminator(batch_fake.X.requires_grad_(True), y=batch_fake.y.requires_grad_(True),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(True))
        
        sp_fake = self.spectral_discriminator(batch_fake.X.requires_grad_(True), y=batch_fake.y.requires_grad_(True),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(True))
        
        
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

        return {'td':loss.item(), 'fd':loss_.item(), 'combined': loss_c.item(), 'out_sp_disc': -sp_fake}

    def set_optimizers(self, optim_discriminator: Optimizer, optim_generator: Optimizer, optim_spectral: Optimizer = None):
        self.optim_discriminator = optim_discriminator
        self.optim_generator = optim_generator
        self.optim_spectral = optim_spectral


