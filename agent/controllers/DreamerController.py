from collections import defaultdict, deque
from copy import deepcopy
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange

from agent.models.DreamerModel import DreamerModel
from agent.models.world_model import MAWorldModel
from agent.models.tokenizer import StateEncoder, StateDecoder, Tokenizer
from agent.models.vq import SimpleVQAutoEncoder, SimpleFSQAutoEncoder
from networks.dreamer.action import Actor,StochasticPolicy
from environments import Env
from utils import discretize_into_bins, bins2continuous, obs_bins2continuous, symexp, symlog, obs_split_into_bins


class DreamerController:

    def __init__(self, config):
        self.config = config
        self.env_type = config.ENV_TYPE

        # tokenizer
        if config.tokenizer_type == 'vq':
            self.tokenizer = SimpleVQAutoEncoder(in_dim=config.IN_DIM, embed_dim=config.EMBED_DIM, num_tokens=config.nums_obs_token,
                                                 codebook_size=config.OBS_VOCAB_SIZE, learnable_codebook=False, ema_update=True).eval()
            self.obs_vocab_size = config.OBS_VOCAB_SIZE
        elif config.tokenizer_type == 'fsq':
            # 2^8 -> [8, 6, 5], 2^10 -> [8, 5, 5, 5]
            self.tokenizer = SimpleFSQAutoEncoder(in_dim=config.IN_DIM, num_tokens=config.nums_obs_token, levels=[8, 6, 5]).eval()
            self.obs_vocab_size = np.prod([8, 6, 5])
        else:
            raise NotImplementedError

        if not config.use_stack:
            if config.CONTINUOUS_ACTION or self.env_type not in [Env.STARCRAFT, Env.SMAX]:
                self.actor = StochasticPolicy(config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
                                              continuous_action=config.CONTINUOUS_ACTION, continuous_action_space=config.ACTION_SPACE)
            else:
                self.actor = Actor(config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS)

        else:
            if config.CONTINUOUS_ACTION or self.env_type not in [Env.STARCRAFT, Env.SMAX]:
                self.actor = StochasticPolicy(config.IN_DIM * config.stack_obs_num, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
                                              continuous_action=config.CONTINUOUS_ACTION, continuous_action_space=config.ACTION_SPACE)
            else:
                self.actor = Actor(config.IN_DIM * config.stack_obs_num, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS)
        
        self.eps = config.epsilon
        
        self.expl_decay = config.EXPL_DECAY
        self.expl_noise = config.EXPL_NOISE
        self.expl_min = config.EXPL_MIN
        self.init_rnns()
        self.init_buffer()

        self.use_bin = config.use_bin
        self.bins = config.bins
        self.temp = config.temperature
        self.use_continuous_action = config.CONTINUOUS_ACTION
        if hasattr(config, 'determinisitc'):
            self.use_deterministic_action = config.determinisitc
        else:
            self.use_deterministic_action = self.use_continuous_action

    def receive_params(self, params):
        self.tokenizer.load_state_dict(params['tokenizer'])
        # self.model.load_state_dict(params['model'])
        self.actor.load_state_dict(params['actor'])

    def init_buffer(self):
        self.buffer = defaultdict(list)

    def init_rnns(self):
        self.prev_rnn_state = None
        self.prev_actions = None
        
        if self.config.use_stack:
            self.stack_obs = deque(maxlen=self.config.stack_obs_num)
            for _ in range(self.config.stack_obs_num):
                self.stack_obs.append(
                    torch.zeros(1, self.config.NUM_AGENTS, self.config.IN_DIM)
                )

    def dispatch_buffer(self):
        total_buffer = {k: np.asarray(v, dtype=np.float32) for k, v in self.buffer.items()}
        last = np.zeros_like(total_buffer['done'])
        last[-1] = 1.0
        total_buffer['last'] = last
        self.init_rnns()
        self.init_buffer()
        return total_buffer

    def update_buffer(self, items):
        for k, v in items.items():
            if v is not None:
                self.buffer[k].append(v.squeeze(0).detach().clone().numpy())
    
    @torch.no_grad()
    def step(self, observations, avail_actions, nn_mask):
        """"
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        # obs_encodings = self.tokenizer.encode(observations, should_preprocess=True).z_quantized
        # feats = rearrange(obs_encodings, 'b n k e -> b n (k e)')
        if not self.use_bin:
            feats = self.tokenizer.encode_decode(observations, True, True)
        else:
            observations = symlog(observations)
            tokens = obs_split_into_bins(observations, self.bins, low=-3., high=3.)
            feats = obs_bins2continuous(tokens, self.bins, low=-3., high=3.)
            feats = symexp(feats)

        if self.config.use_stack:
            self.stack_obs.append(feats)
            feats = rearrange(torch.cat(list(self.stack_obs), dim=0), 'b n e -> 1 n (b e)')

        if self.use_continuous_action:
            action, log_probs = self.actor(feats, deterministic = self.use_deterministic_action)
            ent = self.actor.dist_entropy.clone()
            # if continuous action control, return action_aver_std
            ent = torch.exp(ent / action.shape[-1] - 0.5 - 0.5 * math.log(2 * math.pi))
        else:
            action, pi = self.actor(feats)
            if avail_actions is not None:
                pi[avail_actions == 0] = -1e10  # logits

            pi = pi / self.temp  # softmax temperature
            
            probs = F.softmax(pi, -1)
            ent = -((probs * torch.log(probs + 1e-6)).sum(-1))            
            
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()
            
            # epsilon exploration
            if random.random() < self.eps:
                action_dist = OneHotCategorical(probs=avail_actions / avail_actions.sum(-1, keepdim=True))
                action = action_dist.sample()
        
        return action.squeeze(0).clone(), ent.squeeze(0).clone()

    def advance_rnns(self, state):
        self.prev_rnn_state = deepcopy(state)

    def exploration(self, action):
        """
        :param action: action to take, shape (1,)
        :return: action of the same shape passed in, augmented with some noise
        """
        for i in range(action.shape[0]):
            if np.random.uniform(0, 1) < self.expl_noise:
                index = torch.randint(0, action.shape[-1], (1, ), device=action.device)
                transformed = torch.zeros(action.shape[-1])
                transformed[index] = 1.
                action[i] = transformed
        self.expl_noise *= self.expl_decay
        self.expl_noise = max(self.expl_noise, self.expl_min)
        return action
