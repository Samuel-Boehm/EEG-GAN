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
    

class SpectralTrainer(GanSoftplusTrainer):
    
    def __init__(self, i_logging: int, discriminator: Discriminator, generator: Generator, r1: float,
                 sp_discriminator: Discriminator, sp_optim):
                 
        self.r1_lambda = r1
        self.sp_discriminator = sp_discriminator
        self.sp_optim = sp_optim
        super().__init__(i_logging, discriminator, generator, r1, None)


    def train_discriminator(self, batch_real: Data[torch.Tensor], batch_fake: Data[torch.Tensor], latent: torch.Tensor):
        
        #### Train time domain discriminator ####
        self.discriminator.zero_grad()
        self.optim_discriminator.zero_grad()
        self.discriminator.train(True)


        fx_real_td = self.discriminator(batch_real.X.requires_grad_(True),
                                    y=batch_real.y.requires_grad_(True),
                                    y_onehot=batch_real.y_onehot.requires_grad_(True))
        
        err_real_td = softplus(-fx_real_td).mean()

        fx_fake_td = self.discriminator(batch_fake.X.requires_grad_(True), 
                                    y=batch_fake.y.requires_grad_(True),
                                    y_onehot=batch_fake.y_onehot.requires_grad_(True))

        err_fake_td = softplus(fx_fake_td).mean()
      

        gp_td = self.r1_lambda * gradient_penalty(self.discriminator, batch_real,
                                                batch_fake, batch_real.X.device)
     
        # Calculate error with gradient penalty
        err_td = err_real_td + err_fake_td + gp_td
        err_td.backward()
        
        # Optimize
        self.optim_discriminator.step()

        ##### train frequency domain discriminator ####
        self.sp_discriminator.zero_grad()
        self.sp_optim.zero_grad()
        self.sp_discriminator.train(True)

        fx_real_fd = self.sp_discriminator(batch_real.X.requires_grad_(True))
        
        err_real_fd = softplus(-fx_real_fd).mean()

        fx_fake_fd = self.sp_discriminator(batch_fake.X.requires_grad_(True))

        err_fake_fd = softplus(fx_fake_fd).mean()
    
        gp_fd = self.r1_lambda * gradient_penalty(self.sp_discriminator, batch_real,
                                                batch_fake, batch_real.X.device)
        
        # Calculate error with gradient penalty
        err_fd = err_real_fd + err_fake_fd + gp_fd
        err_fd.backward()
        
        # Optimize
        self.sp_optim.step()
        

        return {"loss_real td": err_real_td.item(), "loss_fake td": err_fake_td.item(),
                "loss_real fd": err_real_fd.item(), "loss_fake fd": err_fake_fd.item(),
                "r1_penalty td": gp_td.item(), "r1_penalty fd": gp_fd.item()}
        

    
    def train_generator(self, batch_real: Data[torch.Tensor]):
        self.generator.zero_grad()
        self.optim_generator.zero_grad()
        self.generator.train(True)

        self.discriminator.train(False)
        self.sp_discriminator.train(False)

        with torch.no_grad():
            latent, y_fake, y_onehot_fake = to_device(batch_real.X.device,
                                                      *self.generator.create_latent_input(self.rng, len(batch_real.X)))
            latent, y_fake, y_onehot_fake = detach_all(latent, y_fake, y_onehot_fake)

        X_fake = self.generator(latent.requires_grad_(False), y=y_fake.requires_grad_(False),
                                y_onehot=y_onehot_fake.requires_grad_(False))
        
        batch_fake = Data[torch.Tensor](X_fake, y_fake, y_onehot_fake)

        # Time domain error
        fx_fake = self.discriminator(batch_fake.X.requires_grad_(True), y=batch_fake.y.requires_grad_(True),
                                     y_onehot=batch_fake.y_onehot.requires_grad_(True))
        
        # Frequency domain error
        fx_fd_fake = self.sp_discriminator(batch_fake.X.requires_grad_(True))
        
        loss1 = softplus(-fx_fake).mean()
        loss2 = softplus(-fx_fd_fake).mean()

        a = 1.0
        b = 1.0 
        loss =  (a*loss1 + b*loss2) / (a+b)
        loss.backward()

        self.optim_generator.step()

        return loss.item()

