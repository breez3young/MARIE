import os
# Force JAX to use CPU to avoid CUDA initialization issues in Ray workers
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper
from gym.spaces import Discrete, Box


class SMAX:
    """
    SMAX environment wrapper that converts JAX-based JaxMARL outputs to numpy arrays
    compatible with PyTorch-based MARIE algorithm.
    """

    def __init__(self, env_name, seed, **kwargs):
        """
        Initialize SMAX environment.
        
        Args:
            env_name: Name of the SMAX scenario (e.g., "5m_vs_6m", "3m", "8m")
            seed: Random seed
            **kwargs: Additional arguments for environment configuration
        """
        self.env_name = env_name
        self.seed = seed
        
        # Parse scenario from env_name
        scenario = map_name_to_scenario(env_name)
        
        # Default configuration
        self.config = {
            'scenario': scenario,
            'use_self_play_reward': kwargs.get('use_self_play_reward', False),
            'walls_cause_death': kwargs.get('walls_cause_death', True),
            'see_enemy_actions': kwargs.get('see_enemy_actions', False),
            'action_type': kwargs.get('action_type', 'discrete'),  # 'discrete' or 'continuous'
            'observation_type': kwargs.get('observation_type', 'unit_list'),  # 'unit_list' or 'conic'
        }
        
        # Create JAX environment with SMAXLogWrapper for proper episode statistics
        base_env = make("HeuristicEnemySMAX", **self.config)
        # base_env = make("SMAX", **self.config)  # no heuristics
        self.env = SMAXLogWrapper(base_env, replace_info=True)
        
        # Initialize random key
        self.key = jax.random.PRNGKey(seed)
        self.key, key_reset = jax.random.split(self.key)
        
        # Get initial observation to determine shapes
        obs, self.state = self.env.reset(key_reset)
        
        # Environment info
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        
        # Get observation and action spaces
        sample_agent = self.agents[0]
        obs_space = self.env.observation_space(sample_agent)
        action_space = self.env.action_space(sample_agent)
        
        # Set dimensions
        if hasattr(obs_space, 'shape'):
            self.n_obs = int(np.prod(obs_space.shape))
        else:
            self.n_obs = obs_space.n
            
        if hasattr(action_space, 'n'):
            self.n_actions = action_space.n
            self.discrete = True
            self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]
            self.individual_action_space = Discrete(self.n_actions)
        else:
            self.n_actions = int(np.prod(action_space.shape))
            self.discrete = False
            self.action_space = [Box(low=action_space.low, high=action_space.high, 
                                    shape=action_space.shape, dtype=np.float32) 
                                for _ in range(self.n_agents)]
            self.individual_action_space = Box(low=action_space.low, high=action_space.high,
                                              shape=action_space.shape, dtype=np.float32)
        
        self.cur_step = 0
        self.max_steps = self.env.max_steps
        self.max_time_steps = self.env.max_steps
        
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            obs_dict: Dictionary of observations {agent_id: obs_array}
        """
        self.key, key_reset = jax.random.split(self.key)
        obs, self.state = self.env.reset(key_reset)
        self.cur_step = 0
        
        # Convert JAX arrays to numpy and create dict with integer keys
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            obs_array = np.array(obs[agent])
            # Flatten if needed
            if len(obs_array.shape) > 1:
                obs_array = obs_array.flatten()
            obs_dict[i] = obs_array
            
        return obs_dict
    
    def step(self, action_dict):
        """
        Take a step in the environment.
        
        Args:
            action_dict: Dictionary of actions {agent_id: action} or array of actions
            
        Returns:
            obs_dict: Dictionary of observations
            reward_dict: Dictionary of rewards
            done_dict: Dictionary of done flags
            info_dict: Dictionary of info
        """
        # Convert action_dict to agent-keyed dict if needed
        if isinstance(action_dict, (list, np.ndarray)):
            actions = {}
            for i, agent in enumerate(self.agents):
                if self.discrete:
                    actions[agent] = jnp.array(int(action_dict[i]))
                else:
                    actions[agent] = jnp.array(action_dict[i])
        else:
            actions = {}
            for i, agent in enumerate(self.agents):
                if i in action_dict:
                    if self.discrete:
                        actions[agent] = jnp.array(int(action_dict[i]))
                    else:
                        actions[agent] = jnp.array(action_dict[i])
        
        # Step environment
        self.key, key_step = jax.random.split(self.key)
        obs, self.state, rewards, dones, infos = self.env.step(key_step, self.state, actions)
        self.cur_step += 1
        
        # Convert to numpy and create dicts with integer keys
        obs_dict = {}
        reward_dict = {}
        done_dict = {}
        info_dict = {}
        
        for i, agent in enumerate(self.agents):
            # Observations
            obs_array = np.array(obs[agent])
            if len(obs_array.shape) > 1:
                obs_array = obs_array.flatten()
            obs_dict[i] = obs_array
            
            # Rewards - use mean reward as in reference code
            agent_reward = float(rewards[agent])
            reward_dict[i] = agent_reward
            
            # Dones
            done_dict[i] = bool(dones[agent])
            
            # Info - individual agent info
            info_dict[i] = {}
            if agent in infos:
                for key, value in infos[agent].items():
                    if isinstance(value, jnp.ndarray):
                        info_dict[i][key] = np.array(value)
                    else:
                        info_dict[i][key] = value
        
        # Episode done check
        episode_done = dones.get("__all__", False) or self.cur_step >= self.max_steps
        if episode_done:
            for i in range(self.n_agents):
                done_dict[i] = True
        
        # SMAXLogWrapper provides these statistics in infos when episode is done
        # Extract them similar to reference code
        if episode_done and 'returned_won_episode' in infos:
            info_dict['battle_won'] = float(infos['returned_won_episode'].any())
            info_dict['episode_return'] = float(infos['returned_episode_returns'].mean())
            info_dict['episode_length'] = int(infos['returned_episode_lengths'].max())
        
        return obs_dict, reward_dict, done_dict, info_dict
    
    def get_avail_actions(self):
        """
        Get available actions for all agents.
        
        Returns:
            List of available actions for each agent
        """
        if not self.discrete:
            return None
            
        avail_actions = []
        try:
            # SMAXLogWrapper wraps the state, access env_state
            state_to_use = self.state.env_state if hasattr(self.state, 'env_state') else self.state
            avail_actions_dict = self.env.get_avail_actions(state_to_use)
            for agent in self.agents:
                if agent in avail_actions_dict:
                    avail = np.array(avail_actions_dict[agent])
                    avail_actions.append(avail.tolist())
                else:
                    # Default: all actions available
                    avail_actions.append([1] * self.n_actions)
        except:
            # If get_avail_actions not available, assume all actions are available
            for _ in range(self.n_agents):
                avail_actions.append([1] * self.n_actions)
                
        return avail_actions
    
    def get_avail_agent_actions(self, agent_id):
        """
        Get available actions for a specific agent.
        
        Args:
            agent_id: Integer agent ID
            
        Returns:
            List of available actions (1 for available, 0 for unavailable)
        """
        if not self.discrete:
            return None
            
        try:
            # SMAXLogWrapper wraps the state, access env_state
            state_to_use = self.state.env_state if hasattr(self.state, 'env_state') else self.state
            avail_actions_dict = self.env.get_avail_actions(state_to_use)
            agent = self.agents[agent_id]
            if agent in avail_actions_dict:
                return np.array(avail_actions_dict[agent]).tolist()
        except:
            pass
            
        # Default: all actions available
        return [1] * self.n_actions
    
    def render(self):
        """Render the environment (if supported)."""
        # SMAX/JaxMARL doesn't have built-in render, but we can print state
        print(f"Step: {self.cur_step}")
        pass
    
    def close(self):
        """Close the environment."""
        # JAX environments don't need explicit closing
        pass
    
    def seed(self, seed):
        """Set random seed."""
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
