import torch.nn as nn
import numpy as np
import torch
from pytorch_lightning.core import LightningModule
from torch.nn.functional import softplus
from eeggan.examples.high_gamma.models.layers.pixelnorm import PixelNorm
from torch.nn.init import calculate_gain
from torch import autograd



class WeightScale(object):
    """
    Implemented for PyTorch using WeightNorm implementation
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html

    References
    ----------
    Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).
    Progressive Growing of GANs for Improved Quality, Stability,
    and Variation. Retrieved from http://arxiv.org/abs/1710.10196
    """

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        w = getattr(module, self.name + '_unscaled')
        c = getattr(module, self.name + '_c')
        tmp = c * w
        return tmp

    @staticmethod
    def apply(module, name, gain):
        fn = WeightScale(name)
        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        # Constant from He et al. 2015

        c = gain / np.sqrt(np.prod(list(weight.size())[1:]))
        setattr(module, name + '_c', float(c))
        module.register_parameter(name + '_unscaled', nn.Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))
        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_unscaled']
        del module._parameters[self.name + '_c']
        module.register_parameter(self.name, nn.Parameter(weight.data))

    def __call__(self, module, inputs, **kwargs):
        setattr(module, self.name, self.compute_weight(module))


def WS(module, gain=calculate_gain('leaky_relu'), name='weight'):
    """
    Helper function that applies equalized learning rate to weights.
    This is used to make the code more readable

    Parameters
    ----------
    module : module
        Module scaling should be applied to (Conv/Linear)
    gain : float
        Gain of following activation layer
        See torch.nn.init.calculate_gain. 
        Defaults to 'leaky_relu' 
    """
    WeightScale.apply(module, name, gain)
    return module

class Generator(nn.Module):
    """
    Description
    ----------
    Generator module for implementing progressive GANs

    Attributes
    ----------
    blocks : list
        List of `GeneratorStage`s. Each represent one
        stage during progression
    n_classes: int
        number of classes, required to create embedding layer 
        for conditional GAN
    """

    def __init__(self, blocks, n_classes, embedding_dim):
        super(Generator, self).__init__()
        # noinspection PyTypeChecker
        self.blocks = nn.ModuleList(blocks)
        self.cur_stage = 0
        self.alpha = 1.

        self.label_embedding = nn.Embedding(n_classes, embedding_dim)

    def forward(self, x, y, **kwargs):
        embedding = self.label_embedding(y)
        #embedding shape: batch_size x 10
        x = torch.cat([x, embedding], dim=1)


        for i in range(0, self.cur_stage + 1):
            x = self.blocks[i](x, last=(i == self.cur_stage), **kwargs)
        
        return x


    def upsample_to_stage(self, x, stage):
        """
        Scales up input to the size of current input stage.
        Utilizes `ProgressiveGeneratorBlock.fade_sequence` from each stage.

        Parameters
        ----------
        x : tensor
            Input data
        stage : int
            Stage to which input should be upwnsampled

        Returns
        -------
        output : tensor
            Upsampled data
        """
        raise NotImplementedError
    
class Critic(nn.Module):
    """
    Critic module for implementing progressive GANs

    Attributes
    ----------
    blocks : list
        List of `CriticStage`s. Each represent one
        stage during progression
    
    Parameters
    ----------
    blocks : list
        List of `ProgressiveDiscriminatorBlock` which each represent one
        stage during progression
    """

    def __init__(self, n_time, n_channels, n_classes, blocks):
        super(Critic, self).__init__()
        # noinspection PyTypeChecker
        self.blocks = nn.ModuleList(blocks)
        self.cur_block = len(self.blocks) - 1
        self.alpha = 1.
        self.n_time = n_time

        self.label_embedding = nn.Embedding(n_classes, n_time)



    def forward(self, x, y, **kwargs):
        
        embedding = self.label_embedding(y).view(y.shape[0], 1, self.n_time)

        embedding = self.downsample_to_stage(embedding, self.cur_block)

        x = torch.cat([x, embedding], 1) # batch_size x n_channels + 1 x n_time 

        for i in range(self.cur_block, len(self.blocks)):
            x = self.blocks[i](x,  first=(i == self.cur_block))
        return x
    
    def downsample_to_stage(self, x, stage):
        """
        Scales down input to the size of current input stage.
    
        Parameters
        ----------
        x : tensor
            Input data
        stage : int
            Stage to which input should be downsampled

        Returns
        -------
        output : tensor
            Downsampled data
        """
        raise NotImplementedError

class GeneratorStage(nn.Module):
    """
    Description
    ----------
    Single stage for the progressive growing generator.
    Each stage consists of two possible actions:
    
    convolution_sequence: data always runs through this block 
    
    out_sequence: if the current stage is the last stage, data 
    gets passed through here. This is needed since the number of
    filter might not match the number of channels in the data



    Attributes
    ----------
    convolution_sequence : nn.Sequence
        Sequence of modules that process stage

    out_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current output
    """

    def __init__(self, intermediate_sequence, out_sequence):
        super(GeneratorStage, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.out_sequence = out_sequence
    

    def forward(self, x, last=False, **kwargs):
        out = self.intermediate_sequence(x, **kwargs)
        if last:
            out = self.out_sequence(out, **kwargs)
        return out


class CriticStage(nn.Module):
    """
    Description
    ----------
    Single stage for the progressive growing critic.
    Each stage consists of two possible actions:
    
    convolution_sequence: data always runs through this block 
    
    in_sequence: if the current stage is the first stage, data 
    gets passed through here. This is needed since the number of
    filters might not match the number of channels in the data
    and therefore a convolution is used to match the number of
    filters. 



    Attributes
    ----------
    convolution_sequence : nn.Sequence
        Sequence of modules that process stage

    in_sequence : nn.Sequence
        Sequence of modules that is applied if stage is the current input
    """

    def __init__(self, intermediate_sequence, out_sequence):
        super(CriticStage, self).__init__()
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = out_sequence
    

    def forward(self, x, first=False, **kwargs):
        
        if first:
            out = self.in_sequence(out, **kwargs)
        
        out = self.intermediate_sequence(x, **kwargs)
        return out


    

class ConvBlockStage(nn.Module):
    '''
    Description
    ----------
    Convolution block for each critic and generator stage. 
    Since the generator uses pixel norm and the critic does 
    not, the 'generator' argument can be used to toggle
    pixel norm on and off. The 'generator' argument also
    sets if the resampling layer is in the beginning or the end.

    Arguments:
    n_filters: int
        number of filters (convolution kernels) used
    stage: int
        progressive stage for which the block is build
    generator: bool
        toggle pixel norm on and off

    '''
    def __init__(self, n_filters,  stage, generator=False):
        super(ConvBlockStage, self).__init__()
        kernel_size =  int(3 + (stage*4)) # stage0: 3, stage1: 7, stage2: 11 ....
        padding = int(stage*2 + 1) # stage0: 1, stage1: 3, stage2, 4 ...
        stride = 1 # fixed to 1 for now
        groups = int(n_filters / ((stage + 1)* 2)) # for n_filters = 120: 60, 30, 20, 15, 12, 10

        self.generator = generator

        self.conv1 = WS(nn.Conv1d(n_filters, n_filters, groups=groups, kernel_size=kernel_size, stride=stride, padding=padding))
        self.conv2 = WS(nn.Conv1d(n_filters, n_filters, groups=groups, kernel_size=kernel_size + 2, stride=stride, padding=padding + 1))
        self.conv3 = WS(nn.Conv1d(n_filters, n_filters, groups=n_filters, kernel_size=1, stride=stride, padding=0))
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.resample = WS(nn.ConvTranspose1d(n_filters, n_filters, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        x = self.resample(x) if self.generator else x # If the block is inside the generator we need to upsample here
        x = self.leaky(x) if self.generator else x # activation for generator 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.pn(x) if self.generator else x
        x = self.conv3(x)
        x = self.leaky(x)
        x = self.pn(x) if self.generator else x
        x = x if  self.generator else self.resample(x) # If the block is inside the critic we need to upsample here
        x = x if self.generator else self.leaky(x) # activation for critic
        return x


class GAN(LightningModule):
    def __init__(
        self,
        n_channels,
        n_classes,
        n_time,
        n_stages,
        n_filters,
        latent_dim: int = 100,
        lambda_gp = 10,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.embedding_dim = 10
        self.current_stage = 0

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
    def forward(self, z, y):
        return self.generator(z, y)
    
    def training_step(self, batch):
        X_real, y_real = batch

        optimizer_g, optimizer_c = self.optimizers()

        # generate noise and labels:
        z = torch.randn(X_real.shape[0], self.hparams.latent_dim)
        z = z.type_as(X_real)

        # generate fake batch:
        y_fake = torch.randint(low=0, high=self.hparams.n_classes, size=X_real.shape[0], dtype=torch.int32)
        X_fake = self.forward(z, y_fake)


        # train critic
        self.toggle_optimizer(optimizer_c)
        
        fx_real = self.critic(X_real, y_real)
        fx_fake = self.critic(X_fake.detach(), y_fake)
        gp = self.gradient_penalty(X_real, X_fake)

        loss_critic = (
                -(torch.mean(fx_real) - torch.mean(fx_fake))
                + self.hparams.lambda_gp * gp
                + (0.001 * torch.mean(fx_real ** 2))
            )

        self.log("c_loss_real", loss_critic, prog_bar=True)
        self.manual_backward(loss_critic)

        optimizer_c.step()
        optimizer_c.zero_grad()
        self.untoggle_optimizer(optimizer_c)


        # train generator
        self.toggle_optimizer(optimizer_g)

        
        fx_fake = self.critic(X_fake, y_fake)
        g_loss = softplus(-fx_fake).mean()
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_c], []
    

    def calc_gradient_penalty(self, batch_real, batch_fake):
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
        alpha = torch.rand(batch_real.data.size(0),*((len(batch_real.data.size())-1)*[1]))
        alpha = alpha.expand(batch_real.data.size())

        interpolates = alpha * batch_real.data + ((1 - alpha) * batch_fake.data)

        critic_interpolates = self.critic(interpolates)

        ones = torch.ones(critic_interpolates.size())

        gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                    grad_outputs=ones,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        tmp = (gradients.norm(2, dim=1) - 1)

        gradient_penalty = ((tmp) ** 2).mean()

        return gradient_penalty



def build_generator(latent_dim, embedding_dim, n_filters, n_time, n_stages, n_channels, n_classes) -> Generator:
    
    
    # Generator:
    n_time_first_layer = int(np.floor(n_time / 2 ** n_stages))
    blocks = []

    # Note that the first conv stage in the generator differs from the others
    # because it takes the latent vector as input
    first_conv = nn.Sequential(
        nn.Linear(latent_dim + embedding_dim, n_filters * n_time_first_layer),
        nn.LeakyReLU(0.2),
        PixelNorm(),
        nn.Unflatten(1, (n_filters, n_time_first_layer)),
        WS(nn.Conv1d(n_filters, n_filters, kernel_size=4, padding=1, stride=1)),
        ConvBlockStage(n_filters, 0, generator=True),
        nn.ConvTranspose1d(n_filters, n_filters, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2)
        )
    
    generator_out = WS(nn.Conv1d(n_filters, n_channels, kernel_size=1, padding=1, stride=1))

    blocks.append(GeneratorStage(first_conv, generator_out))

    for stage in range(1, n_stages):
        stage_conv = ConvBlockStage(n_filters, stage, generator=True)

        # Out sequence is independent of stage
        blocks.append(GeneratorStage(stage_conv, generator_out))

    return Generator(blocks, n_classes, embedding_dim)

def build_critic(n_filters, n_time, n_stages, n_channels, n_classes):

    n_time_first_layer = int(np.floor(n_time / 2 ** n_stages))
    
    # Critic:
    blocks = []
    
    last_conv = nn.Sequential(
        ConvBlockStage(n_filters, 0, generator=False),
        nn.Flatten(),
        nn.Linear(n_filters * n_time_first_layer, 1),
    )

    critic_in = nn.Sequential(
        WS(nn.Conv1d(n_channels, n_filters, kernel_size=1, stride=1)),
        nn.LeakyReLU(0.2),
    )

    blocks.append(CriticStage(last_conv, critic_in))

    for stage in range(1, n_stages):
        stage_conv = ConvBlockStage(n_filters, stage, generator=False)

        # In sequence is independent of stage
        blocks.append(CriticStage(stage_conv, critic_in))
    
    return Critic(n_time, n_channels, n_classes, blocks)