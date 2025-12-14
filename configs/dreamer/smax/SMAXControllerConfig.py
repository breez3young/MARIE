from agent.controllers.DreamerController import DreamerController
from configs.dreamer.smax.SMAXAgentConfig import SMAXDreamerConfig


class SMAXDreamerControllerConfig(SMAXDreamerConfig):
    def __init__(self):
        super().__init__()

        self.epsilon = 0.  # Epsilon for exploration
        self.EXPL_DECAY = 0.9999
        self.EXPL_NOISE = 0.
        self.EXPL_MIN = 0.
        
        self.temperature = 1.  # Temperature for action sampling

    def create_controller(self):
        return DreamerController(self)
