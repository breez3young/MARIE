import argparse
import os
import shutil
import datetime
from pathlib import Path

from agent.runners.DreamerRunner import DreamerRunner
from configs import Experiment
from configs.EnvConfigs import EnvCurriculumConfig, StarCraftConfig, PettingZooConfig, FootballConfig, MAMujocoConfig, SMAXConfig


from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
# for MPE
from configs.dreamer.mpe.MpeLearnerConfig import MPEDreamerLearnerConfig
from configs.dreamer.mpe.MpeControllerConfig import MPEDreamerControllerConfig

# for GRF
from configs.dreamer.football.GRFLearnerConfig import GRFDreamerLearnerConfig
from configs.dreamer.football.GRFControllerConfig import GRFDreamerControllerConfig

# for MAMuJoCo
from configs.dreamer.mamujoco.mamujocoLearnerConfig import MAMujocoDreamerLearnerConfig
from configs.dreamer.mamujoco.mamujocoControllerConfig import MAMujocoDreamerControllerConfig

# for SMAX
from configs.dreamer.smax.SMAXLearnerConfig import SMAXDreamerLearnerConfig
from configs.dreamer.smax.SMAXControllerConfig import SMAXDreamerControllerConfig


from environments import Env
from utils import generate_group_name, format_numel_str_deci

import torch
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="flatland", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')

    # specialized arg for MAMujoco
    parser.add_argument('--agent_conf', type=str, default=None)

    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--seed', type=int, default=1, help='Number of workers')
    parser.add_argument('--steps', type=int, default=1e6, help='Number of workers')
    parser.add_argument('--mode', type=str, default='disabled')
    parser.add_argument('--tokenizer', type=str, default='vq')
    parser.add_argument('--decay', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1.)  # for controller sampling data

    parser.add_argument('--sample_temp', type=float, default='inf')

    parser.add_argument('--average_r', action='store_true')
    parser.add_argument('--ce_for_r', action='store_true')
    parser.add_argument('--ce_for_av', action='store_true')
    parser.add_argument('--ce_for_end', action='store_true')

    return parser.parse_args()


def train_dreamer(exp, n_workers): 
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes, save_interval = 200000, save_mode = 'interval')


def get_env_info(configs, env):
    if not env.discrete:
        assert hasattr(env, 'individual_action_space')
        individual_action_space = env.individual_action_space
    else:
        individual_action_space = None

    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = individual_action_space
    
    print(f'Observation dims: {env.n_obs}')
    print(f'Action dims: {env.n_actions}')
    print(f'Num agents: {env.n_agents}')
    print(f'Continuous action for control? -> {not env.discrete}')
    
    if hasattr(env, 'individual_action_space'):
        print(f'Individual action space: {env.individual_action_space}')

    env.close()


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 2000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_pettingzoo_configs(env_name, continuous_action = True):
    agent_configs = [MPEDreamerControllerConfig(), MPEDreamerLearnerConfig()]
    env_config = PettingZooConfig(env_name, RANDOM_SEED, continuous_action)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_football_configs(env_name):
    agent_configs = [GRFDreamerControllerConfig(), GRFDreamerLearnerConfig()]
    env_config = FootballConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_mamujoco_configs(scenario, agent_config):
    agent_configs = [MAMujocoDreamerControllerConfig(), MAMujocoDreamerLearnerConfig()]
    env_config = MAMujocoConfig(scenario = scenario, seed = RANDOM_SEED, agent_conf = agent_config)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_smax_configs(env_name):
    agent_configs = [SMAXDreamerControllerConfig(), SMAXDreamerLearnerConfig()]
    env_config = SMAXConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

if __name__ == "__main__":
    RANDOM_SEED = 23
    args = parse_args()
    RANDOM_SEED += args.seed * 100
    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.SMAX:
        configs = prepare_smax_configs(args.env_name)
    elif args.env == Env.PETTINGZOO:
        configs = prepare_pettingzoo_configs(args.env_name, continuous_action=True)
    elif args.env == Env.GRF:
        configs = prepare_football_configs(args.env_name)
    elif args.env == Env.MAMUJOCO:
        configs = prepare_mamujoco_configs(args.env_name, args.agent_conf)
    else:
        raise Exception("Unknown environment")
    
    # seed everywhere
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    torch.autograd.set_detect_anomaly(True)
    # --------------------
    
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)
    
    configs["learner_config"].seed = RANDOM_SEED

    configs["learner_config"].tokenizer_type = args.tokenizer
    configs["controller_config"].tokenizer_type = args.tokenizer
    configs["learner_config"].ema_decay = args.decay
    configs["controller_config"].ema_decay = args.decay
    
    configs["controller_config"].temperature = args.temperature

    configs["learner_config"].critic_average_r = args.average_r

    configs["learner_config"].use_ce_for_r = args.ce_for_r
    configs["learner_config"].use_ce_for_end = False  # args.ce_for_end
    configs["learner_config"].use_ce_for_av_action = args.ce_for_av

    rewards_prediction_config = configs["learner_config"].rewards_prediction_config

    if args.sample_temp == float('inf'):
        configs["learner_config"].sample_temperature = str(args.sample_temp)
    else:
        configs["learner_config"].sample_temperature = args.sample_temp

    current_date = datetime.datetime.now()
    current_date_string = current_date.strftime("%m%d")
    # current_date_string = "extreme_partial"

    # make run directory
    dir_prefix = args.env_name + '-'+ args.agent_conf if args.agent_conf is not None else args.env_name

    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/{current_date_string}_results") / args.env / (dir_prefix + f"-{args.tokenizer}")
    # curr_run = f"run{random.randint(1000, 9999)}"
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        os.makedirs(str(run_dir / "ckpt"))

    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "agent"), dst=run_dir / "agent")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "configs"), dst=run_dir / "configs")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "networks"), dst=run_dir / "networks")
    shutil.copyfile(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "train.py"), dst=run_dir / "train.py")
    
    print(f"Run files are saved at {str(run_dir)}\n")
    # -------------------

    configs["learner_config"].RUN_DIR = str(run_dir)
    configs["learner_config"].map_name = args.env_name

    group_name = generate_group_name(args, configs["learner_config"])
    postfix = "_reward-average" if args.average_r else ""
    postfix += f'_sample_temp={args.sample_temp}' if not configs["learner_config"].CONTINUOUS_ACTION else ""
    
    # postfix += "_use_id" if configs["learner_config"].observe_agent_id else ""
    # postfix += "_use_last_action" if configs["learner_config"].observe_last_action else ""
    # postfix += "_use_feat_norm" if configs["learner_config"].use_feature_norm else ""

    if configs["learner_config"].use_ce_for_r:
        run_name = f'(t_embed={configs["learner_config"].EMBED_DIM}) marie_{args.env_name}_{args.agent_conf}_seed_{RANDOM_SEED}_' + format_numel_str_deci(args.steps) + f'_interval={configs["learner_config"].N_SAMPLES}_{rewards_prediction_config["loss_type"]}_bins{rewards_prediction_config["bins"]}' + postfix
    else:
        run_name = f'(t_embed={configs["learner_config"].EMBED_DIM}) marie_{args.env_name}_{args.agent_conf}_seed_{RANDOM_SEED}_' + format_numel_str_deci(args.steps) + f'_interval={configs["learner_config"].N_SAMPLES}' + postfix

    prefix = f"({current_date_string}_T={args.temperature}_eval-T=1.0)" if not configs["learner_config"].CONTINUOUS_ACTION else f"({current_date_string})"

    global wandb
    import wandb
    wandb.init(
        project=args.env,
        mode=args.mode,
        group=prefix + group_name,
        name=run_name,
        config=configs["learner_config"].to_dict(),
        notes="",
    )

    exp = Experiment(steps=args.steps,
                     episodes=50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_dreamer(exp, n_workers=args.n_workers)
