import sys
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict
from einops import rearrange
from torch.utils.data.dataloader import DataLoader

import numpy as np
import torch

from agent.memory.DreamerMemory import DreamerMemory, ObsDataset
from agent.models.DreamerModel import DreamerModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout, trans_actor_rollout, continuous_actor_loss
from agent.optim.loss import huber_loss, mse_loss
from agent.optim.utils import advantage
from agent.utils.valuenorm import ValueNorm
from environments import Env
from networks.dreamer.action import Actor, StochasticPolicy
from networks.dreamer.critic import AugmentedCritic, Critic, FeatureNormedAugmentedCritic

from agent.models.tokenizer import Tokenizer, StateDecoder, StateEncoder
from agent.models.vq import SimpleVQAutoEncoder, SimpleFSQAutoEncoder
from agent.models.world_model import MAWorldModel
from utils import configure_optimizer
from episode import SC2Episode, MpeEpisode, GRFEpisode, MamujocoEpisode
from dataset import MultiAgentEpisodesDataset

import wandb
import ipdb

def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:

    def __init__(self, config):
        self.config = config
        self.env_type = config.ENV_TYPE
        # self.config.update()

        torch.autograd.set_detect_anomaly(True)
        
        # tokenizer
        if self.config.tokenizer_type == 'vq':
            self.tokenizer = SimpleVQAutoEncoder(in_dim=config.IN_DIM, embed_dim=config.EMBED_DIM, num_tokens=config.nums_obs_token,
                                                 codebook_size=config.OBS_VOCAB_SIZE, learnable_codebook=False, ema_update=True, decay=config.ema_decay).to(config.DEVICE).eval()
            self.obs_vocab_size = config.OBS_VOCAB_SIZE
        elif self.config.tokenizer_type == 'fsq':
            # 2^8 -> [8, 6, 5], 2^10 -> [8, 5, 5, 5]
            levels = [8, 8, 8]
            self.tokenizer = SimpleFSQAutoEncoder(in_dim=config.IN_DIM, num_tokens=config.nums_obs_token, levels=levels).to(config.DEVICE).eval()
            self.obs_vocab_size = np.prod(levels)
        else:
            raise NotImplementedError
        # ---------

        # world model (transformer)
        obs_vocab_size = config.bins if config.use_bin else self.obs_vocab_size
        perattn_config = config.perattn_config(num_latents=config.NUM_AGENTS)
        
        ## --------------update--------------
        num_action_token = 1 if not config.CONTINUOUS_ACTION else config.ACTION_SIZE
        num_obs_token = config.IN_DIM if config.use_bin else config.nums_obs_token
        act_vocab_size = config.ACTION_SIZE if not config.CONTINUOUS_ACTION else config.action_bins
        combine_action = False # (config.ENV_TYPE == Env.MAMUJOCO)
        if combine_action:
            num_action_token = num_action_token * config.NUM_AGENTS

        transformer_config = config.trans_config(tokens_per_block=num_obs_token + num_action_token + 1)  # 最后一个1是aggregated token的数量
        action_low = None if not config.CONTINUOUS_ACTION else config.ACTION_SPACE.low.min()
        action_high = None if not config.CONTINUOUS_ACTION else config.ACTION_SPACE.high.max()
        ## ----------------------------------

        self.model = MAWorldModel(obs_vocab_size=obs_vocab_size, act_vocab_size=act_vocab_size, num_action_tokens=num_action_token, num_agents=config.NUM_AGENTS,
                                  config=transformer_config, perattn_config=perattn_config, action_dim=config.ACTION_SIZE,
                                  ### used for bins (no tokenizer)
                                  use_bin=config.use_bin, bins=config.bins,
                                  ### used for continuous action discretization
                                  action_bins = config.action_bins, action_low=action_low, action_high=action_high, combine_action = combine_action,
                                  ### used for setting the prediction head
                                  use_symlog=False, use_ce_for_end=config.use_ce_for_end, use_ce_for_av_action=config.use_ce_for_av_action, enable_av_pred=(config.ENV_TYPE in [Env.STARCRAFT, Env.SMAX]),
                                  use_ce_for_reward=config.use_ce_for_r, rewards_prediction_config=config.rewards_prediction_config).to(config.DEVICE).eval()
        # -------------------------

        # based on latent
        # self.actor = Actor(config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
        # self.critic = AugmentedCritic(config.critic_FEAT, config.HIDDEN).to(config.DEVICE)

        # based on reconstructed obs
        if not self.config.use_stack:
            if config.CONTINUOUS_ACTION or self.env_type not in [Env.STARCRAFT, Env.SMAX]:
                print(f"Use continuous action policy.")
                self.actor = StochasticPolicy(config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
                                              continuous_action=config.CONTINUOUS_ACTION, continuous_action_space=config.ACTION_SPACE).to(config.DEVICE)
                self.critic = FeatureNormedAugmentedCritic(config.IN_DIM, config.ACTION_HIDDEN, config.ACTION_LAYERS, feat_norm=True).to(config.DEVICE)
                self.value_normalizer = ValueNorm(1, device=config.DEVICE)
            else:
                self.actor = Actor(config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
                self.critic = AugmentedCritic(config.IN_DIM, config.HIDDEN).to(config.DEVICE)
        
        else:
            print(f"Use stacking observation mode. Currently stack {config.stack_obs_num} observations for decision making.")
            if config.CONTINUOUS_ACTION or self.env_type not in [Env.STARCRAFT, Env.SMAX]:
                print(f"Use continuous action policy.")
                self.actor = StochasticPolicy(config.IN_DIM * config.stack_obs_num, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
                                              continuous_action=config.CONTINUOUS_ACTION, continuous_action_space=config.ACTION_SPACE).to(config.DEVICE)
                self.critic = FeatureNormedAugmentedCritic(config.IN_DIM * config.stack_obs_num, config.ACTION_HIDDEN, config.ACTION_LAYERS, feat_norm=True).to(config.DEVICE)
                self.value_normalizer = ValueNorm(1, device=config.DEVICE)
            else:
                self.actor = Actor(config.IN_DIM * config.stack_obs_num, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
                self.critic = AugmentedCritic(config.IN_DIM * config.stack_obs_num, config.HIDDEN).to(config.DEVICE)

        if not config.CONTINUOUS_ACTION and self.env_type in [Env.STARCRAFT, Env.SMAX]:
            initialize_weights(self.actor)
            initialize_weights(self.critic, mode='xavier')

        self.old_critic = deepcopy(self.critic)
        
        self.replay_buffer = MultiAgentEpisodesDataset(max_ram_usage="30G", name="train_dataset", temp=20)
        self.mamba_replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, config.NUM_AGENTS,
                                                 config.DEVICE, config.ENV_TYPE, config.sample_temperature)
        
        ## (debug) pre-load mamba training buffer
        if self.config.is_preload:
            print(f"Load offline dataset from {self.config.load_path}")
            # self.replay_buffer.load_from_pkl(self.config.load_path)
            self.mamba_replay_buffer.load_from_pkl(self.config.load_path)

        self.entropy = config.ENTROPY
        self.step_count = -1
        self.train_count = 0
        self.cur_wandb_epoch = 0
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        self.tqdm_vis = True
        self.use_valuenorm = config.use_valuenorm
        self.use_huber_loss = config.use_huber_loss
        self.use_clipped_value_loss = config.use_clipped_value_loss

        if self.config.use_bin:
            print('Disable using & training tokenizer...')

        if self.config.critic_average_r:
            print("Enable average mode for predicted rewards...")
        else:
            print("Disable average mode for predicted rewards...")

    def init_optimizers(self):
        # self.tokenizer_optimizer = torch.optim.Adam(self.tokenizer.parameters(), lr=self.config.t_lr)
        self.tokenizer_optimizer = torch.optim.AdamW(self.tokenizer.parameters(), lr=3e-4)

        self.model_optimizer = configure_optimizer(self.model, self.config.wm_lr, self.config.wm_weight_decay)
        # self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR, weight_decay=1e-4)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR,
                                                 weight_decay=0.0 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 0.00001,
                                                 eps=1e-5 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR,
                                                 weight_decay=0.0 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 0.00001,
                                                 eps=1e-5 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 1e-8)

    def params(self):
        return {'tokenizer': {k: v.cpu() for k, v in self.tokenizer.state_dict().items()},
                'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}
        
    def load_pretrained_wm(self, load_path):
        ckpt = torch.load(load_path)
        self.tokenizer.load_state_dict(ckpt['tokenizer'])
        self.model.load_state_dict(ckpt['model'])
        
        self.tokenizer.eval()
        self.model.eval()

    def save(self, save_path):
        torch.save(self.params(), save_path)

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])

        self.add_experience_to_dataset(rollout)
        self.mamba_replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                        rollout['fake'], rollout['last'], rollout.get('avail_action'))
        # self.vq_buffer.append(rollout['observation'])
        
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.mamba_replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        self.train_count += 1

        intermediate_losses = defaultdict(float)
        # train tokenzier
        if not self.config.use_bin:
            pbar = tqdm(range(self.config.MODEL_EPOCHS), desc=f"Training tokenizer", file=sys.stdout, disable=not self.tqdm_vis)
            for _ in pbar:
                samples = self.mamba_replay_buffer.sample_batch(bs=256, sl=1, mode="tokenizer")
                samples = self._to_device(samples)

                # loss_dict = self.train_tokenizer(samples)
                if self.config.tokenizer_type == 'vq':
                    loss_dict = self.train_vq_tokenizer(samples['observation'])

                    pbar.set_description(
                        f"Training tokenizer:"
                        + f"rec loss: {loss_dict[self.config.tokenizer_type + '/rec_loss']:.4f} | "
                        + f"cmt loss: {loss_dict[self.config.tokenizer_type + '/cmt_loss']:.4f} | "
                        + f"active %: {loss_dict[self.config.tokenizer_type + '/active']:.3f} | "
                    )
                elif self.config.tokenizer_type == 'fsq':
                    loss_dict = self.train_fsq_tokenizer(samples['observation'])

                    pbar.set_description(
                        f"Training tokenizer:"
                        + f"rec loss: {loss_dict[self.config.tokenizer_type + '/rec_loss']:.4f} | "
                        + f"active %: {loss_dict[self.config.tokenizer_type + '/active']:.3f} | "
                    )
                else:
                    raise NotImplementedError

                for loss_name, loss_value in loss_dict.items():
                    intermediate_losses[loss_name] += loss_value / self.config.MODEL_EPOCHS

        if self.train_count == 10:
            print('Start training world model...')
        # 9 5
        if (self.train_count > 9 and not self.config.use_bin) or (self.train_count > 5 and self.config.use_bin):
            # train transformer-based world model
            pbar = tqdm(range(self.config.WM_EPOCHS), desc=f"Training {str(self.model)}", file=sys.stdout, disable=not self.tqdm_vis)
            for _ in pbar:
                samples = self.mamba_replay_buffer.sample_batch(bs=self.config.MODEL_BATCH_SIZE, sl=self.config.HORIZON, mode="model")
                samples = self._to_device(samples)
                attn_mask = self.mamba_replay_buffer.generate_attn_mask(samples["done"], self.model.config.tokens_per_block).to(self.config.DEVICE)

                loss_dict = self.train_model(samples, attn_mask)

                for loss_name, loss_value in loss_dict.items():
                    intermediate_losses[loss_name] += loss_value / self.config.WM_EPOCHS

                pbar.set_description(
                    f"Training world model:"
                    + f"total loss: {loss_dict['world_model/total_loss']:.4f} | "
                    + f"obs loss: {loss_dict['world_model/loss_obs']:.4f} | "
                    + f"rew loss: {loss_dict['world_model/loss_rewards']:.4f} | "
                    + f"dis loss: {loss_dict['world_model/loss_ends']:.3f} | "
                    + f"av loss: {loss_dict['world_model/loss_av_actions']:.3f} | "
                )

        if self.train_count == 20:
            print('Start training actor & critic...')

        if self.train_count > 19:
            # train actor-critic
            for i in tqdm(range(self.config.EPOCHS), desc=f"Training actor-critic", file=sys.stdout, disable=not self.tqdm_vis):
                samples = self.replay_buffer.sample_batch(batch_num_samples=self.config.ac_batch_size, # self.config.MODEL_BATCH_SIZE * 2
                                                          sequence_length=self.config.stack_obs_num if self.config.use_stack else 1,
                                                          sample_from_start=False,
                                                          valid_sample=False)
                
                # samples = self.mamba_replay_buffer.sample_batch(bs=30, sl=20)

                samples = self._to_device(samples)
                self.train_agent_with_transformer(samples)

        wandb.log({'epoch': self.cur_wandb_epoch, **intermediate_losses})
        
        if self.train_count % 200 == 0 and self.train_count > 19 and False:
            self.model.eval()
            self.tokenizer.eval()
            sample = self.replay_buffer.sample_batch(batch_num_samples=1,
                                                     sequence_length=self.config.HORIZON,
                                                     sample_from_start=True,
                                                     valid_sample=True)
            sample = self._to_device(sample)
            self.model.visualize_attn(sample, self.tokenizer, Path(self.config.RUN_DIR) / "visualization" / "attn" / f"epoch_{self.train_count}")
        
        self.cur_wandb_epoch += 1
        
    def visualize_attention_map(self, epoch, save_mode='interval'):
        if save_mode == 'interval':
            save_path = Path(self.config.RUN_DIR) / "visualization" / "attn" / f"epoch_{epoch}"
        elif save_mode == 'final':
            save_path = Path(self.config.RUN_DIR) / "visualization" / "attn" / "final"
        
        self.model.eval()
        self.tokenizer.eval()
        sample = self.replay_buffer.sample_batch(batch_num_samples=1,
                                                    sequence_length=self.config.HORIZON,
                                                    sample_from_start=True,
                                                    valid_sample=True)
        sample = self._to_device(sample)
        self.model.visualize_attn(sample, self.tokenizer, save_path)
    
    def train_vq_tokenizer(self, x):
        assert type(self.tokenizer) == SimpleVQAutoEncoder
        self.tokenizer.train()

        out, indices, cmt_loss = self.tokenizer(x, True, True)
        rec_loss = (out - x).abs().mean()
        loss = rec_loss + self.config.alpha * cmt_loss

        active_rate = indices.detach().unique().numel() / self.obs_vocab_size * 100

        self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
        self.tokenizer.eval()

        loss_dict = {
            self.config.tokenizer_type + "/cmt_loss": cmt_loss.item(),
            self.config.tokenizer_type + "/rec_loss": rec_loss.item(),
            self.config.tokenizer_type + "/active": active_rate,
        }

        return loss_dict

    def train_fsq_tokenizer(self, x):
        assert type(self.tokenizer) == SimpleFSQAutoEncoder
        self.tokenizer.train()

        out, indices = self.tokenizer(x, True, True)
        loss = (out - x).abs().mean()

        active_rate = indices.detach().unique().numel() / self.obs_vocab_size * 100

        self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
        self.tokenizer.eval()

        loss_dict = {
            self.config.tokenizer_type + "/rec_loss": loss.item(),
            self.config.tokenizer_type + "/active": active_rate,
        }

        return loss_dict

    # def train_tokenizer(self, samples):
    #     self.tokenizer.train()
    #     loss, loss_dict = self.tokenizer.compute_loss(samples)
    #     self.apply_optimizer(self.tokenizer_optimizer, self.tokenizer, loss, self.config.max_grad_norm)
    #     self.tokenizer.eval()
    #     return loss_dict
    
    def train_model(self, samples, attn_mask = None):
        self.tokenizer.eval()
        self.model.train()
        
        # loss, loss_dict = self.model.compute_loss(samples, self.tokenizer)
        loss, loss_dict = self.model.compute_loss(samples, self.tokenizer, attn_mask)
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.max_grad_norm) # or GRAD_CLIP
        self.model.eval()
        return loss_dict

    def train_agent_with_transformer(self, samples):
        self.tokenizer.eval()
        self.model.eval()
        actions, av_actions, old_policy, actor_feat, critic_feat, returns, old_values \
              = trans_actor_rollout(samples['observation'],  # rearrange(samples['observation'], 'b l n e -> (b l) 1 n e'),
                                    samples['av_action'] if 'av_action' in samples else None,  # rearrange(samples['av_action'], 'b l n e -> (b l) 1 n e'),
                                    samples['filled'], # samples['last']
                                    self.tokenizer, self.model,
                                    self.actor,
                                    self.critic if self.env_type != Env.PETTINGZOO else self.old_critic, # self.critic
                                    self.config,
                                    env_type=self.env_type,
                                    external_rew_model=None,
                                    use_stack=self.config.use_stack,
                                    stack_obs_num=self.config.stack_obs_num if self.config.use_stack else None,
                                    use_valuenorm = self.use_valuenorm,
                                    value_normalizer = self.value_normalizer if self.use_valuenorm else None,
                                    )
        
        if self.use_valuenorm:
            B, N = old_values.shape[:-1]
            unnormalized_old_v = self.value_normalizer.denormalize(rearrange(old_values, 'b n 1 -> (b n) 1'))
            unnormalized_old_v = rearrange(unnormalized_old_v, '(b n) 1 -> b n 1', b=B, n=N)
            adv = returns.detach() - unnormalized_old_v
        else:
            adv = returns.detach() - self.critic(critic_feat).detach()


        if self.env_type in [Env.STARCRAFT, Env.GRF, Env.MAMUJOCO, Env.SMAX]:
            adv = advantage(adv)
        elif self.env_type == Env.PETTINGZOO:
            pass
        else:
            raise NotImplementedError('Error')

        wandb.log({'Agent/Returns': returns.mean()})
        
        self.cur_update += 1
        
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])

            step = 2000
            if self.env_type in [Env.MAMUJOCO]:
                # if environment is MAMujoco, we set the step according to the num_mini_batch
                step = int(len(inds) / self.config.num_mini_batch)

            for i in range(0, len(inds), step):
                idx = inds[i:i + step]

                if not self.config.CONTINUOUS_ACTION:
                    loss = actor_loss(actor_feat[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                      old_policy[idx], adv[idx], self.actor, self.entropy)
                else:
                    loss = continuous_actor_loss(actor_feat[idx], actions[idx], None,
                                                 old_policy[idx], adv[idx], self.actor, self.entropy, self.config.clip_param)
                
                actor_grad_norm = self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING

                # using value normalization
                if self.use_valuenorm:
                    # get new values
                    values = self.critic(critic_feat[idx])
                    
                    value_pred_clipped = old_values[idx] + (values - old_values[idx]).clamp(
                        - self.config.clip_param, self.config.clip_param
                    )

                    bb, nn = returns[idx].shape[:-1]
                    self.value_normalizer.update(rearrange(returns[idx], 'b n 1 -> (b n) 1'))
                    normalized_returns = rearrange(self.value_normalizer.normalize(rearrange(returns[idx], 'b n 1 -> (b n) 1')),
                                                   '(b n) 1 -> b n 1', b=bb, n=nn)
                    error_clipped  = normalized_returns.clone() - value_pred_clipped
                    error_original = normalized_returns.clone() - values

                    if self.use_huber_loss:
                        value_loss_clipped = huber_loss(error_clipped, self.config.huber_delta)
                        value_loss_original = huber_loss(error_original, self.config.huber_delta)
                    else:
                        value_loss_clipped = mse_loss(error_clipped)
                        value_loss_original = mse_loss(error_original)

                    if self.use_clipped_value_loss:
                        val_loss = torch.max(value_loss_original, value_loss_clipped)
                    else:
                        val_loss = value_loss_original

                    val_loss = val_loss.mean()

                else:
                    val_loss = value_loss(self.critic, critic_feat[idx], returns[idx])

                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                critic_grad_norm = self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                
                wandb.log({'Agent/actor_grad_norm': actor_grad_norm, 'Agent/critic_grad_norm': critic_grad_norm})
        
        # hard update critic
        if self.cur_update % self.config.TARGET_UPDATE == 0:
            self.old_critic = deepcopy(self.critic)
            self.cur_update = 0

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        return grad_norm

    ## add data to dataset
    def add_experience_to_dataset(self, data):
        if self.env_type == Env.STARCRAFT or self.env_type == Env.SMAX:
            episode = SC2Episode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                av_action=torch.FloatTensor(data['avail_action'].copy()) if 'avail_action' in data else None,   # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )
        elif self.env_type == Env.PETTINGZOO:
            episode = MpeEpisode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )
        elif self.env_type == Env.GRF:
            episode = GRFEpisode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )

        elif self.env_type == Env.MAMUJOCO:
            episode = MamujocoEpisode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )
        
        else:
            raise NotImplementedError

        self.replay_buffer.add_episode(episode)

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.config.DEVICE) if batch[k] is not None else None for k in batch}
