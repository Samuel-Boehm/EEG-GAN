# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>


from lightning.pytorch.callbacks import Callback
from gan.model.gan import GAN
from lightning.pytorch import Trainer

class Scheduler(Callback):

    def __init__(self, fading_period:int, **kwargs):
        '''
        Scheduler for the progression of the GAN.
        Parameters:
        -----------
            fading_period: int, number of epochs for fading between stages.
        '''

        super().__init__()
        self.fading_period = fading_period

    
    def on_train_epoch_start(self, trainer: Trainer, model: GAN):
        '''
        Each epoch start, check and inizialize the next stage if necessary.
        '''
        
        # increase alpha after each epoch
        model.generator.alpha += 1/self.fading_period
        model.critic.alpha += 1/self.fading_period
        
        if trainer.current_epoch in model.progression_epochs:
            trainer.datamodule.set_stage(model.current_stage)
            model.generator.set_stage(model.current_stage)
            model.critic.set_stage(model.current_stage)
            model.sp_critic.set_stage(model.current_stage)

            # increase stage
            model.current_stage += 1 
