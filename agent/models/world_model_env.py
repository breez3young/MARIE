import random
from typing import List, Optional, Union

import gym
from einops import rearrange, repeat
import numpy as np
from PIL import Image
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torchvision

from utils import action_split_into_bins, discretize_into_bins, bins2continuous, symexp, obs_bins2continuous, symlog, obs_split_into_bins
from environments import Env

import ipdb


class MAWorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env_type: str, env: Optional[gym.Env] = None) -> None:
        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()

        ####
        self.use_bin = self.world_model.use_bin
        self.bins = self.world_model.bins
        ####

        self.tokenizer = tokenizer.to(self.device).eval() if not self.use_bin else None

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env
        self.env_type = env_type
        ## necessary params
        self.mode = 'ar' if world_model.config.attention == 'causal' else 'bg'
        self.n_agents = world_model.num_agents
        self.predict_avail_action = world_model.enable_av_pred

        self.use_continuous_action = world_model.use_continuous_action
        self.action_low = world_model.action_low
        self.action_high = world_model.action_high
        self.action_bins = world_model.action_bins
        self.combine_action = world_model.combine_action

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    ## unmodified
    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        if self.use_bin:
            observations = symlog(observations)
            obs_tokens = obs_split_into_bins(observations, self.bins, low=-3., high=3.)
            
        else:
            # obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, N, Obs_dim) -> (B, N, K)
            _, obs_tokens = self.tokenizer.encode(observations, should_preprocess=True)

        num_observations_tokens = obs_tokens.shape[-1]
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        output_sequence = self.refresh_keys_values_with_initial_obs_tokens(rearrange(obs_tokens, 'b n k -> (b n) k'))
        self.obs_tokens = obs_tokens
        critic_feat = rearrange(output_sequence[:, -1], '(b n) k -> b n k', b=int(output_sequence.size(0) / self.n_agents), n=self.n_agents)
        return self.decode_obs_tokens(), critic_feat

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape # (B, K)
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.Tensor], should_predict_next_obs: bool = True):
        if type(action) == np.ndarray:
            action = torch.tensor(action, device=self.device)

        if self.mode == 'ar':
            return self.step_ar(action=action, should_predict_next_obs=should_predict_next_obs)
        elif self.mode == 'bg':
            return self.step_bg(action=action, should_predict_next_obs=should_predict_next_obs)
        else:
            raise ValueError(f'Mode {self.mode} has no corresponding step!')

    ### TODO
    @torch.no_grad()
    def step_bg(self, action: Union[List, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True):
        raise NotImplementedError

    # 
    @torch.no_grad()
    def step_ar(self, action: Union[List, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True):
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        ## update
        if self.use_continuous_action:
            valid_actions = torch.clip(action, self.action_low, self.action_high)
            action = action_split_into_bins(valid_actions, self.action_bins, self.action_low, self.action_high)
            if self.combine_action:
                combined_action = rearrange(action, 'b n a -> b (n a)')
                combined_action = repeat(combined_action, 'b a -> b n a', n=self.n_agents)

        # perceiver attention output
        perattn_out = self.world_model.get_perceiver_attn_out(self.obs_tokens, action)
        perattn_out = rearrange(perattn_out, 'b n e -> (b n) 1 e')
        # ---------------------------

        if self.combine_action:
            token = combined_action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(combined_action, dtype=torch.long).clone().detach()
        else:
            token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long).clone().detach()

        perattn_placeholder = torch.empty(*token.shape[:-1], 1, dtype=torch.long, device=token.device)
        token = torch.cat([token, perattn_placeholder], dim=-1)

        token = rearrange(token, 'b n k -> (b n) k').to(self.device)  # (B, N)

        for k in range(num_passes):  # assumption that there is only one action token.
            outputs_wm = self.world_model(token, perattn_out=perattn_out, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)
            
            if k == 0:

                if not self.world_model.use_ce_for_reward:
                    reward = outputs_wm.pred_rewards.float().squeeze(-2)

                else:
                    probs = F.softmax(outputs_wm.pred_rewards, dim=-1)
                    reward = self.world_model.reward_loss.transform_from_probs(probs)
                
                if self.world_model.use_symlog:
                    reward = symexp(reward)
                
                # done = Categorical(logits=outputs_wm.logits_ends).sample().unsqueeze(-1).to(torch.bool)  # (B,), numpy
                if not self.world_model.use_ce_for_end:
                    pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs_wm.logits_ends), 1)
                    done = pred_ends.mean
                else:
                    raise NotImplementedError
                    done = Categorical(logits=outputs_wm.logits_ends).sample().unsqueeze(-2)

                if self.predict_avail_action:

                    if not self.world_model.use_ce_for_av_action:
                        avail_action_dist = td.independent.Independent(td.Bernoulli(logits=outputs_wm.pred_avail_action), 1)
                        avail_action = avail_action_dist.sample()
                    else:
                        avail_action_dist = Categorical(logits=outputs_wm.pred_avail_action)
                        avail_action = avail_action_dist.sample()

                else:
                    avail_action = None

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)
            
            perattn_out = None

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        obs_tokens = torch.cat(obs_tokens, dim=1)             # (B, K)

        self.obs_tokens = rearrange(obs_tokens, '(b n) k -> b n k', b=int(obs_tokens.size(0) / self.n_agents), n=self.n_agents)
        
        reward = rearrange(reward, '(b n) 1 -> b n 1', b=int(obs_tokens.size(0) / self.n_agents), n=self.n_agents)
        # reward = reward.squeeze(1)
        
        done = rearrange(done, '(b n) 1 1 -> b n 1', b=int(obs_tokens.size(0) / self.n_agents), n=self.n_agents)
        # done = done.squeeze(1)
        
        avail_action = rearrange(avail_action, '(b n) 1 e -> b n e', b=int(obs_tokens.size(0) / self.n_agents), n=self.n_agents) if avail_action is not None else None
        # avail_action = avail_action.squeeze(1) if avail_action is not None else None
        
        obs = self.decode_obs_tokens() if should_predict_next_obs else None # obs is tensor
        critic_feat = rearrange(output_sequence[:, -1], '(b n) k -> b n k', b=int(obs_tokens.size(0) / self.n_agents), n=self.n_agents) if should_predict_next_obs else None
        # share_obs = self.world_model.get_perceiver_attn_out(self.tokenizer.embedding(self.obs_tokens))

        return obs, reward, done, avail_action, critic_feat # o_t+1, r_t

    ## unmodified
    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self):
        # assert self.obs_tokens.shape[0] % self.n_agents == 0
        # bs = self.obs_tokens.shape[0]
        if not self.use_bin:
            # embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
            # rec = self.tokenizer.decode(embedded_tokens, should_postprocess=True)
            # rec = rearrange(rec, '(b n) o -> b n o', b=int(bs / self.n_agents), n=self.n_agents)
            
            # rec has been clamped into [-1, 1] during the decoding process
            rec = self.tokenizer.decode(self.obs_tokens, should_postprocess=True)

            if self.env_type in [Env.STARCRAFT, Env.SMAX]:
                return rec.clamp(-1., 1.)
            elif self.env_type == Env.PETTINGZOO:
                return rec
            elif self.env_type == Env.GRF:
                return rec
            elif self.env_type == "maniskill2":
                return rec
            
            elif self.env_type == Env.MAMUJOCO:
                return rec

            else:
                raise ValueError(f'Unsupported env {self.env_type}')
        else:
            rec = obs_bins2continuous(self.obs_tokens, self.bins, low=-3., high=3.)
            # inverse transform using symexp
            rec = symexp(rec)
            return rec

    ## unmodified
    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]