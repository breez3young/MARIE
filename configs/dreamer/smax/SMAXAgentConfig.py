from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

from agent.models.tokenizer import StateEncoderConfig
from configs.dreamer.DreamerAgentConfig import DreamerConfig

from functools import partial

RSSM_STATE_MODE = 'discrete'


class SMAXDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.use_valuenorm = False # True
        self.use_huber_loss = False # True
        self.use_clipped_value_loss = False # True
        self.huber_delta = 10.0

        self.use_bin = False
        self.bins = 512
        self.action_bins = 256

        ## debug
        self.use_stack = False
        self.stack_obs_num = 5
