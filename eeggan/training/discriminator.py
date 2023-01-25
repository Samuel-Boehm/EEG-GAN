#  Author: Kay Hartmann <kg.hartma@gmail.com>

from abc import ABCMeta

from eeggan.pytorch.modules.module import Module


class Discriminator(Module, metaclass=ABCMeta):
    """
    Base descriminator Class
    """
    def __init__(self, n_samples, n_channels, n_classes):
        super().__init__()
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_classes = n_classes
    
    def downsample_to_block(self, x, i_block):
        """
        Scales down input to the size of current input stage.
        Utilizes `ProgressiveDiscriminatorBlock.fade_sequence` from each stage.

        Parameters
        ----------
        x : autograd.Variable
            Input data
        i_block : int
            Stage to which input should be downsampled

        Returns
        -------
        output : autograd.Variable
            Downsampled data
        """
        for i in range(i_block):
            x = self.blocks[i].fade_sequence(x)
        output = x
        return output
