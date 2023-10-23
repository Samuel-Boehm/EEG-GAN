# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>


from lightning.pytorch.callbacks import Callback



class Scheduler(Callback):

    def __init__(self, fading_period, **kwargs):
        super().__init__()
        self.fading_period = fading_period

    
    def on_train_epoch_start(self, trainer, model):
        '''
        Each epoch start, check if we need to increase stage. If so, increase stage.
        '''
        
        # increase alpha after each epoch
        model.generator.alpha += 1/self.fading_period
        model.critic.alpha += 1/self.fading_period
        
        if trainer.current_epoch in model.progression_epochs:
            # set stage in data, critic and generator
            trainer.datamodule.set_stage(model.current_stage)
            model.generator.set_stage(model.current_stage)
            model.critic.set_stage(model.current_stage)
            model.sp_critic.set_stage(model.current_stage)

            # increase stage
            model.current_stage += 1 
