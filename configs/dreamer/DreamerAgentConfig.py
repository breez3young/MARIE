from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

from agent.models.tokenizer import StateEncoderConfig
from agent.models.transformer import PerceiverConfig, TransformerConfig

from functools import partial

RSSM_STATE_MODE = 'discrete'


class DreamerConfig(Config):
    def __init__(self):
        super().__init__()
        self.LOG_FOLDER = 'wandb/'

        # optimal smac config
        self.HIDDEN = 256
        self.MODEL_HIDDEN = 256
        self.EMBED = 256
        self.N_CATEGORICALS = 32
        self.N_CLASSES = 32
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        self.DETERMINISTIC = 256
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 256
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 256
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 2
        self.ACTION_HIDDEN = 256
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 256
        self.GAMMA = 0.99  # discount factor
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95  # lambda in dreamer v2
        self.IN_DIM = 30

        self.num_mini_batch = 1
        self.use_valuenorm = False
        self.use_huber_loss = False
        self.use_clipped_value_loss = False
        self.huber_delta = 10.0

        ## discretize params
        self.use_bin = False
        self.bins = 512
        self.action_bins = 256

        # tokenizer params
        self.nums_obs_token = 16 # 4
        self.hidden_sizes = [512, 512]
        self.alpha = 1.0
        self.EMBED_DIM = 128 # 128
        self.OBS_VOCAB_SIZE = 512 # 512

        self.alpha = 10.
        self.ema_decay = 0.8

        self.encoder_config_fn = partial(StateEncoderConfig,
            nums_obs_token=self.nums_obs_token, 
            hidden_sizes=self.hidden_sizes,
            alpha=1.0,
            z_channels=self.EMBED_DIM * self.nums_obs_token
        )
        
        # world model params
        self.HORIZON = 15  # 15
        self.TRANS_EMBED_DIM = 256 # 256
        self.HEADS = 4
        self.perattn_HEADS = 4
        self.DROPOUT = 0.1
        
        # lack "num_latents" which should be equal to NUM_AGENTS
        self.perattn_config = partial(PerceiverConfig,
            dim=self.TRANS_EMBED_DIM,
            latent_dim=self.TRANS_EMBED_DIM,
            depth=2,
            cross_heads=8, # 1
            cross_dim_head=64,
            latent_heads=8,
            latent_dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        self.trans_config = partial(TransformerConfig,
            # tokens_per_block=self.nums_obs_token + 1 + 1,
            max_blocks=self.HORIZON,
            attention='causal',
            num_layers=10, # 10
            num_heads=self.HEADS,
            embed_dim=self.TRANS_EMBED_DIM,
            embed_pdrop=self.DROPOUT,
            resid_pdrop=self.DROPOUT,
            attn_pdrop=self.DROPOUT,
        )

        # 这里修改一下
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        # self.FEAT = self.EMBED_DIM * self.nums_obs_token
        self.critic_FEAT = self.TRANS_EMBED_DIM # * self.nums_obs_token # self.TRANS_EMBED_DIM
        self.GLOBAL_FEAT = self.FEAT + self.EMBED

        ## debug
        self.use_stack = True
        self.stack_obs_num = 5


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]
