from copy import deepcopy

import ray
import torch
from collections import defaultdict

from environments import Env
import numpy as np
import ipdb

@ray.remote
class DreamerWorker:

    def __init__(self, idx, env_config, controller_config):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller()
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config.ENV_TYPE

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0
        elif self.env_type == Env.SMAX:
            return self.done[handle] == 0
        elif self.env_type == Env.PETTINGZOO:
            return self.done[handle] == 0
        elif self.env_type == Env.GRF:
            return self.done[handle] == 0
        elif self.env_type == Env.MAMUJOCO:
            return self.done[handle] == 0
        else:
            raise NotImplementedError(f"{self.env_type} is currently not supported.")

    def _select_actions(self, state):
        avail_actions = []
        observations = []
        fakes = []
        nn_mask = None

        for handle in range(self.env.n_agents):
            if self.env_type == Env.STARCRAFT or self.env_type == Env.SMAX:
                avail_actions.append(torch.tensor(self.env.get_avail_agent_actions(handle)))

            if self._check_handle(handle) and handle in state:
                fakes.append(torch.zeros(1, 1))
                observations.append(state[handle].unsqueeze(0))
            elif self.done[handle] == 1:
                fakes.append(torch.ones(1, 1))
                observations.append(self.get_absorbing_state())
            else:
                fakes.append(torch.zeros(1, 1))
                observations.append(torch.zeros(1, self.in_dim))

        observations = torch.cat(observations).unsqueeze(0)
        av_action = torch.stack(avail_actions).unsqueeze(0) if len(avail_actions) > 0 else None
        actions, ent = self.controller.step(observations, av_action, nn_mask)
        return actions, observations, torch.cat(fakes).unsqueeze(0), av_action, ent

    def _wrap(self, d):
        for key, value in d.items():
            d[key] = torch.tensor(value).float()
        return d
    
    def unwrap(self, d):
        l = []
        for k, v in d.items():
            l.append(v)
        return l

    def get_absorbing_state(self):
        state = torch.zeros(1, self.in_dim)
        return state

    def augment(self, data, inverse=False):
        aug = []
        default = list(data.values())[0].reshape(1, -1)
        for handle in range(self.env.n_agents):
            if handle in data.keys():
                aug.append(data[handle].reshape(1, -1))
            else:
                aug.append(torch.ones_like(default) if inverse else torch.zeros_like(default))
        return torch.cat(aug).unsqueeze(0)

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT:
            return "episode_limit" not in info
        else:
            return steps_done < self.env.max_time_steps

    def run(self, dreamer_params):
        self.controller.receive_params(dreamer_params)
        if self.env_type == Env.STARCRAFT or self.env_type == Env.SMAX:
            state = self._wrap(self.env.reset())
        elif self.env_type == Env.PETTINGZOO:
            state, shared_obs, _ = self.env.reset()
            state = self._wrap(state)
        elif self.env_type == Env.GRF:
            state, shared_obs, _ = self.env.reset()
            state = self._wrap(state)
        elif self.env_type == Env.MAMUJOCO:
            state, shared_obs, _ = self.env.reset()
            state = self._wrap(state)
        else:
            raise NotImplementedError(f'Currently we do not support {self.env_type} env.')
            
        steps_done = 0
        self.done = defaultdict(lambda: False)

        rewards_list = []
        info_list = []

        while True:
            steps_done += 1
            actions, obs, fakes, av_actions, ent = self._select_actions(state)
            if self.env_type == Env.STARCRAFT or self.env_type == Env.SMAX:
                next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions)])
            elif self.env_type == Env.PETTINGZOO:
                next_state, shared_obs, reward, done, info, _ = self.env.step(actions)
                rewards_list.append(np.array(self.unwrap(reward)))
            elif self.env_type == Env.GRF:
                next_state, shared_obs, reward, done, info, _ = self.env.step([action.argmax().item() for i, action in enumerate(actions)])
                info_list.append(info)
            
            elif self.env_type == Env.MAMUJOCO:
                next_state, shared_obs, reward, done, info, _ = self.env.step(actions)
                rewards_list.append(np.array(self.unwrap(reward)))
                info_list.append(info)

            else:
                raise NotImplementedError(f"{self.env_type} is currently not supported.")

            next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
            self.done = done
            self.controller.update_buffer({"action": actions,
                                           "observation": obs,
                                           "reward": self.augment(reward),
                                           "done": self.augment(done),
                                           "fake": fakes,
                                           "avail_action": av_actions,
                                           "entropy": ent, # newly added
                                           })

            state = next_state
            if all([done[key] == 1 for key in range(self.env.n_agents)]):
                # if self._check_termination(info, steps_done):
                #     obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                #     actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                #     index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                #     actions.scatter_(2, index.unsqueeze(-1), 1.)
                #     items = {"observation": obs,
                #              "action": actions,
                #              "reward": torch.zeros(1, self.env.n_agents, 1),
                #              "fake": torch.ones(1, self.env.n_agents, 1),
                #              "done": torch.ones(1, self.env.n_agents, 1),
                #              "avail_action": torch.ones_like(actions) if self.env_type == Env.STARCRAFT else None}
                #     self.controller.update_buffer(items)
                #     self.controller.update_buffer(items)

                break

        if self.env_type == Env.STARCRAFT or self.env_type == Env.SMAX:
            reward = 1. if 'battle_won' in info and info['battle_won'] else 0.
        elif self.env_type == Env.PETTINGZOO:
            rew_per_step = np.mean(rewards_list)
            reward = rew_per_step

        elif self.env_type == Env.GRF:
            reward = self.check_score(info_list)

        elif self.env_type == Env.MAMUJOCO:
            rew_per_step = np.mean(rewards_list)
            reward = rew_per_step
        
        return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "reward": reward,
                                                   "steps_done": steps_done}
    
    def check_score(self, info_list):
        score_reward = 0.
        for info in info_list:
            # take agent 0 for example
            score_reward += info[0]['score_reward']
        
        return score_reward
