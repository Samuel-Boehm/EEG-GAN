from generator import Generator
from critic import Critic

class ProgressionHandler:
    def __init__(self, generator: Generator, critic:Critic, n_stages, epochs_per_stage) -> None:
        self.critic = critic
        self.generator = generator
        self.n_stages = n_stages
        self.epochs_per_stage = epochs_per_stage
        self.epochs_total:int = int(n_stages * epochs_per_stage)

    def set_stage(self, stage):
        self.generator.set_stage(stage)
        self.critic.set_stage(stage)

