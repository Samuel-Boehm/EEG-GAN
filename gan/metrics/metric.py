# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from abc import abstractmethod
from lightning.pytorch import Trainer, LightningModule
from gan.handler.LoggingHandler import batch_data
from typing import Any

class Metric:
    """
    Base class used to build new metric callbacks. Each metric needs to define a __call__ method.
    The call method requires arguments trainer, pl_module and batch.
    **kwargs can be used to pass additional arguments to the metric.
    
    Constant variables can be set when initializing the metric. If the __init__ method is overwritten, 
    make sure to call super().__init__(**kwargs) to pass the kwargs to the base class.

    Args:
        every_n_epochs (int): The metric is calculated every n epochs. If set to 0, the metric
        is calculated once at the end of the training. Defaults to 0.
    """

    def __init__(self, every_n_epochs:int = 0) -> Any:
        """ Initializes the metric. """
        self.interval = every_n_epochs
    
    def __call__(self, trainer: Trainer, module: LightningModule, batch: batch_data) -> Any:
        """ Returns the metric value as a dictionary. """

