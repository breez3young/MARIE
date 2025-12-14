import numpy as np
import torch
import pickle
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, random_split
# from torch.utils.data.dataloader import DataLoader

from environments import Env



class DreamerMemory:
    def __init__(self, capacity, sequence_length, action_size, obs_size, n_agents, device, env_type, sample_temperature = 'inf'):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.action_size = action_size
        self.obs_size = obs_size
        self.device = device
        self.env_type = env_type
        self.sample_temperature = sample_temperature

        self.init_buffer(n_agents, env_type)

    def init_buffer(self, n_agents, env_type):
        self.observations = np.empty((self.capacity, n_agents, self.obs_size), dtype=np.float32)
        self.actions = np.empty((self.capacity, n_agents, self.action_size), dtype=np.float32)
        self.av_actions = np.empty((self.capacity, n_agents, self.action_size),
                                   dtype=np.float32) if env_type in [Env.STARCRAFT, Env.SMAX] else None
        self.rewards = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.fake = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.last = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.next_idx = 0  # 这个next_idx 就是长度
        self.size = 0

        self.sample_visits = {
            "tokenizer": torch.zeros(self.capacity, dtype=torch.long),
            "model": torch.zeros(self.capacity, dtype=torch.long)
        }
        self.n_agents = n_agents
        self.full = False

    def append(self, obs, action, reward, done, fake, last, av_action):
        if self.actions.shape[-2] != action.shape[-2]:
            self.init_buffer(action.shape[-2], self.env_type)
        for i in range(len(obs)):
            self.observations[self.next_idx] = obs[i]
            self.actions[self.next_idx] = action[i]
            if av_action is not None:
                self.av_actions[self.next_idx] = av_action[i]
            self.rewards[self.next_idx] = reward[i]
            self.dones[self.next_idx] = done[i]
            self.fake[self.next_idx] = fake[i]
            self.last[self.next_idx] = last[i]

            for k in self.sample_visits.keys():
                self.sample_visits[k][self.next_idx] = 0

            self.next_idx = (self.next_idx + 1) % self.capacity
            self.size += 1
            self.full = self.full or self.next_idx == 0
            if self.full:
                self.size = self.capacity

    def tenzorify(self, nparray):
        return torch.from_numpy(nparray).float()

    def sample(self, batch_size):
        return self.get_transitions(self.sample_positions(batch_size))

    def process_batch(self, val, idxs, batch_size, s_length = None):
        return torch.as_tensor(val[idxs].reshape(s_length if s_length is not None else self.sequence_length, batch_size, self.n_agents, -1)).to(self.device)

    def get_transitions(self, idxs):
        batch_size = len(idxs)
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.process_batch(self.observations, vec_idxs, batch_size)[1:]
        reward = self.process_batch(self.rewards, vec_idxs, batch_size)[:-1]
        action = self.process_batch(self.actions, vec_idxs, batch_size)[:-1]
        av_action = self.process_batch(self.av_actions, vec_idxs, batch_size)[1:] if self.env_type in [Env.STARCRAFT, Env.SMAX] else None
        done = self.process_batch(self.dones, vec_idxs, batch_size)[:-1]
        fake = self.process_batch(self.fake, vec_idxs, batch_size)[1:]
        last = self.process_batch(self.last, vec_idxs, batch_size)[1:]

        return {'observation': observation, 'reward': reward, 'action': action, 'done': done, 
                'fake': fake, 'last': last, 'av_action': av_action}
    
    def sample_n(self, batch_size):
        return self.get_transitions_n(self.sample_positions(batch_size))

    def get_transitions_n(self, idxs):
        batch_size = len(idxs)
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.process_batch(self.observations, vec_idxs, batch_size)
        reward = self.process_batch(self.rewards, vec_idxs, batch_size)
        action = self.process_batch(self.actions, vec_idxs, batch_size)
        av_action = self.process_batch(self.av_actions, vec_idxs, batch_size) if self.env_type in [Env.STARCRAFT, Env.SMAX] else None
        done = self.process_batch(self.dones, vec_idxs, batch_size)
        fake = self.process_batch(self.fake, vec_idxs, batch_size)
        last = self.process_batch(self.last, vec_idxs, batch_size)

        return {'observation': observation.transpose(1, 0), 'reward': reward.transpose(1, 0), 'action': action.transpose(1, 0), 'done': done.transpose(1, 0), 
                'fake': fake.transpose(1, 0), 'last': last.transpose(1, 0), 'av_action': av_action.transpose(1, 0)}

    def sample_position(self):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.next_idx - self.sequence_length)
            idxs = np.arange(idx, idx + self.sequence_length) % self.capacity
            valid_idx = self.next_idx not in idxs[1:]  # Make sure data does not cross the memory index
        return idxs

    def sample_positions(self, batch_size):
        return np.asarray([self.sample_position() for _ in range(batch_size)])

    def __len__(self):
        return self.capacity if self.full else self.next_idx

    def clean(self):
        self.memory = list()
        self.position = 0

    ### new
    def generate_attn_mask(self, dones, tokens_per_block):
        b, t, n = dones.shape[:3]
        sequence_length = t * tokens_per_block

        dones = dones.all(-2).reshape(b, t)
    
        mask = torch.zeros(b * n, sequence_length, sequence_length)
        for idx in range(b):
            has_done = dones[idx][:-1].sum()
            if has_done == 0:
                mask[idx * n : (idx + 1) * n] = torch.tril(torch.ones(n, sequence_length, sequence_length))

            else:
                done_idx = (dones[idx] == 1).nonzero().squeeze(-1) + 1
                last_j = 0
                for j in done_idx.tolist():
                    mask[idx * n : (idx + 1) * n, (last_j * tokens_per_block) : (j * tokens_per_block), (last_j * tokens_per_block) : (j * tokens_per_block)] = torch.tril(torch.ones(n, (j - last_j) * tokens_per_block, (j - last_j) * tokens_per_block))
                    last_j = j
                    
                mask[idx * n : (idx + 1) * n, (last_j * tokens_per_block) :, (last_j * tokens_per_block) :] = torch.tril(torch.ones(n, (t - last_j) * tokens_per_block, (t - last_j) * tokens_per_block))
        
        return mask
    
    ### for balanced sampling
    def _compute_visit_probs(self, n, mode="tokenizer"):
        temperature = self.sample_temperature
        if temperature == 'inf':
            visits = self.sample_visits[mode][:n].float()
            visit_sum = visits.sum()
            if visit_sum == 0:
                probs = torch.full_like(visits, 1 / n)
            else:
                probs = 1 - visits / visit_sum
        else:
            logits = self.sample_visits[mode][:n].float() / -temperature
            probs = F.softmax(logits, dim=0)
        assert probs.device.type == 'cpu'
        return probs

    def validate_indices(self, indices, sequence_length):
        idxs = (torch.arange(sequence_length) + indices.unsqueeze(1)) % self.capacity
        valid_indices = indices[(idxs[:, 1:] != self.next_idx).all(-1)]
        return valid_indices

    def sample_indices(self, max_batch_size, sequence_length, mode="tokenizer"):
        n = self.size - sequence_length + 1 if not self.full else self.capacity
        batch_size = max_batch_size
        if batch_size * sequence_length > n:
            raise ValueError('Not enough data in buffer')

        probs = self._compute_visit_probs(n, mode)
        start_idx = torch.multinomial(probs, batch_size, replacement=False)
        
        if sequence_length > 1:
            concat_list = []
            all_valid = False

            valid_indices = self.validate_indices(start_idx, sequence_length)
            concat_list.append(valid_indices)
            if valid_indices.size(0) == batch_size:
                all_valid = True
            else:
                rest_indices_num = batch_size - valid_indices.size(0)

            while not all_valid:
                rest_indices = torch.multinomial(probs, rest_indices_num, replacement=False)
                valid_indices = self.validate_indices(rest_indices, sequence_length)
                concat_list.append(valid_indices)
                if valid_indices.size(0) == rest_indices_num:
                    all_valid = True
                else:
                    rest_indices_num = rest_indices_num - valid_indices.size(0)

            start_idx = torch.cat(concat_list, dim=0).to(torch.long)

        # stay on cpu
        flat_idx = start_idx.reshape(-1)
        flat_idx, counts = torch.unique(flat_idx, return_counts=True)
        self.sample_visits[mode][flat_idx] += counts

        start_idx = start_idx.to(device="cpu")
        idx = (start_idx.unsqueeze(-1) + torch.arange(sequence_length, device="cpu")) % self.capacity
        return idx.numpy()

    def sample_batch(self, bs, sl, mode="tokenizer"):
        idxs = self.sample_indices(bs, sl, mode)
        batch_size = len(idxs)
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.process_batch(self.observations, vec_idxs, batch_size, sl)
        reward = self.process_batch(self.rewards, vec_idxs, batch_size, sl)
        action = self.process_batch(self.actions, vec_idxs, batch_size, sl)
        av_action = self.process_batch(self.av_actions, vec_idxs, batch_size, sl) if self.env_type in [Env.STARCRAFT, Env.SMAX] else None
        done = self.process_batch(self.dones, vec_idxs, batch_size, sl)
        fake = self.process_batch(self.fake, vec_idxs, batch_size, sl)
        last = self.process_batch(self.last, vec_idxs, batch_size, sl)
        
        return {'observation': observation.transpose(1, 0), 'reward': reward.transpose(1, 0), 'action': action.transpose(1, 0), 'done': done.transpose(1, 0), 
                'fake': fake.transpose(1, 0), 'last': last.transpose(1, 0), 'av_action': av_action.transpose(1, 0) if av_action is not None else None}
    
    def load_from_pkl(self, dataset_path, remove_fake=True):
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        
        if remove_fake:
            valid_indices = np.argwhere(data["fakes"].all(-2).squeeze() == False).squeeze().tolist()
            data['observations'] = data['observations'][valid_indices]
            data['actions'] = data['actions'][valid_indices]
            data['av_actions'] = data['av_actions'][valid_indices]
            data['rewards'] = data['rewards'][valid_indices]
            data['dones'] = data['dones'][valid_indices]

        db_size = data['observations'].shape[0]
        assert self.capacity >= db_size
        self.next_idx = db_size

        self.observations[:self.next_idx] = data['observations']
        self.actions[:self.next_idx] = data['actions']
        self.av_actions[:self.next_idx] = data['av_actions']
        self.rewards[:self.next_idx] = data['rewards']
        self.dones[:self.next_idx] = data['dones']
        
        self.size += db_size
        self.next_idx = self.next_idx % self.capacity
        self.full = self.full or self.next_idx == 0
        if self.full:
            self.size = self.capacity


class ObsDataset(Dataset):
    def __init__(self, capacity, obs_size, n_agents):
        self.capacity = capacity
        self.obs_dim = obs_size
        self.n_agents = n_agents

        self.observations = np.empty((self.capacity, n_agents, obs_size), dtype=np.float32)
        self.size = 0
        self.next_idx = 0
        self.full = False
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.observations[index]
    
    def append(self, new_obs):
        for i in range(len(new_obs)):
            self.observations[self.next_idx] = new_obs[i]

            self.next_idx = (self.next_idx + 1) % self.capacity
            self.size += 1
            self.full = self.full or self.next_idx == 0
            if self.full:
                self.size = self.capacity