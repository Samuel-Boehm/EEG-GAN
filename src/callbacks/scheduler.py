# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>


from lightning.pytorch.callbacks import Callback
from lightning.pytorch import Trainer
import numpy as np

from src.models import GAN

from src.utils import RankedLogger 

log = RankedLogger(__name__, rank_zero_only=True)


class Scheduler(Callback):

    def __init__(self, n_fading_epochs:int, epochs_per_stage, n_stages:int, **kwargs):
        '''
        Scheduler for the progression of the GAN.
        Parameters:
        -----------
            n_fading_epochs: int, number of epochs for fading between stages.
            epochs_per_stage: int or list of ints, number of epochs per stage.
            n_stages: int, number of stages.
        '''

        super().__init__()
        self.n_fading_epochs = n_fading_epochs
        self.n_stages = n_stages

        # Determine transition epochs:
        self.progression_epochs = []

        if isinstance(epochs_per_stage, int):
            for i in range(n_stages):
                self.progression_epochs.append(epochs_per_stage*i)
        
        else:
            # try to convert to list
            try:
                epochs_per_stage = list(epochs_per_stage)
            except:
                raise f"epochs_per_stage must be either int or list of ints but got {type(epochs_per_stage)}"
            
            if len(epochs_per_stage) != n_stages:
                raise ValueError (
                f"""
                len(epochs_per_stage) != n_stages. Number of stages must be equal to the
                number of epochs per stage. Got {len(epochs_per_stage)} epochs and {n_stages} stages.
                {epochs_per_stage}
                """)
            else:
                self.progression_epochs = np.cumsum(epochs_per_stage)
                # We need to add a 0 to the beginning of the list, in order to trigger the 
                # 'set_stage' method at the beginning of the training. 

    def on_train_start(self, trainer: Trainer, model:GAN):
        '''
        Set the current stage to 1 at the beginning of the training.
        '''
        model.current_stage = 1
        trainer.datamodule.set_stage(model.current_stage)
        model.generator.set_stage(model.current_stage)
        model.critic.set_stage(model.current_stage)
        model.sp_critic.set_stage(model.current_stage)
        
    def on_train_epoch_end(self, trainer: Trainer, model:GAN):
        '''
        Each epoch end, check and inizialize the next stage if necessary.
        '''
        
        # increase alpha after each epoch
        model.generator.alpha += 1/self.n_fading_epochs
        model.critic.alpha += 1/self.n_fading_epochs
        
        if trainer.current_epoch in self.progression_epochs:
            model.current_stage += 1
            log.info(f"Epoch {trainer.current_epoch} reached - transitioning from stage {model.current_stage -1} to stage {model.current_stage}")
            trainer.datamodule.set_stage(model.current_stage)
            model.generator.set_stage(model.current_stage)
            model.critic.set_stage(model.current_stage)
            model.sp_critic.set_stage(model.current_stage)
