# Project: EEG-GAN
# Author: Samuel Boehm
# E-Mail: <samuel-boehm@web.de>

from lightning import Callback

class Scheduler(Callback):
    def on_train_epoch_start(self, trainer, model):
        
        '''
        Each epoch start, check if we need to increase stage. If so, increase stage.
        '''
        
        if trainer.current_epoch in model.progression_epochs:
            # set stage in data, critic and generator
            trainer.datamodule.set_stage(model.current_stage)
            model.generator.set_stage(model.current_stage)
            model.critic.set_stage(model.current_stage)

            # increase stage
            model.current_stage += 1 
