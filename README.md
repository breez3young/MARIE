# MARIE - Multi-Agent auto-Regressive Imagination for Efficient learning

Code for the paper *Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models*. [Paper link](https://openreview.net/forum?id=xT8BEgXmVc)

## News

🎉 **[2025-12]** MARIE now supports **SMAX** (JAX-based StarCraft Multi-Agent Challenge) with 14 available maps! We provide CPU-only JAX integration to ensure compatibility with Ray distributed workers. See [smax-integration.md](smax-integration.md) for detailed usage.

🎉 **[2025-02]** We open-source the codebase of MARIE.

## Installation

`python3.10+` is required

```bash
pip install wheel

# install torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# install ray for distributed workers
pip install ray==2.7.2

```

We currently implement the interfaces for common MARL environments and they can be used directly.

### Installing SMAC (Main exp)

Please follow [the official instructions](https://github.com/oxwhirl/smac) to install SMAC.

```bash
pip install git+https://github.com/oxwhirl/smac.git
```

### Installing SMAX (StarCraft Multi-Agent Challenge with JAX)

SMAX is a JAX-based StarCraft Multi-Agent Challenge environment via JaxMARL. We use CPU-only JAX to avoid CUDA issues with Ray workers.

```bash
pip install "jax[cpu]==0.4.31"
pip install jaxlib
pip install jaxmarl
```

For detailed SMAX integration information, see [smax-integration.md](smax-integration.md).

### Installing Google Research Football (Deprecated)

Please follow [the official instructions](https://github.com/google-research/football) to install Google Research Football.

### Installing MPE

```
pip install pettingzoo==1.22.2
pip install supersuit==3.7.0
```

### Installing MAMujoco

```
pip install "mujoco-py==2.1.2.14"
pip install "Jinja2==3.0.3"
pip install "glfw==2.5.1"
pip install "Cython==0.29.28"
pip install patchelf
```

Please follow [the instruction on PKU-HARL repo](https://github.com/PKU-MARL/HARL/tree/main) to install MAMujoco.

Encounter any issues with the usage of Mujoco? Refer to [this troubleshooting page](https://pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html).

## Usage

An example for running MARIE on a specific environment

```bash
python3 train.py --n_workers 1 --env <ENV> --env_name <SCENARIO_NAME> --seed <SEED> --agent_conf <AGENT_CONF> \
  --steps <TOTAL_STEPS> --mode <WANDB_LOG_MODE> --tokenizer <TOKENIZER_CLASS> --decay 0.8 \
  --temperature <POLICY_TEMPERATURE> --sample_temp <SAMPLING_TEMPERATURE> --ce_for_av 
```

- ```ENV```: which environment to evaluate MARIE on. Five options: starcraft, smax, football, pettingzoo, mamujoco
- ```SCENARIO_NAME```: which scenario or map to evaluate MARIE on. For example, in SMAC, we can set SCENARIO_NAME as *1c3s5z* to evaluate.
- ```AGENT_CONF```: which agent splitting configure to use in MAMujco. Only valid in running experiments on MAMujoco.
- ```TOTAL_STEPS```: the maximum environment steps in the low data regime.
- ```WANDB_LOG_MODE```: whether to enable wandb logging. Options: disabled, offline, online
- ```TOKENIZER_CLASS```: which tokenizer to use. Options: fsq, vq.
- ```POLICY_TEMPERATURE```: control the exploration of the policy (only useful in the case of discrete action space).
- ```SAMPLING_TEMPERATURE```: control the sample visitation probability during training VQ and World Model. (Refer to the balanced sampling in [TWM](https://github.com/jrobine/twm).)

Across all experiments in our work, we always set SAMPLING_TEMPERATURE as 'inf', i.e., standard sampling without any consideration of potential unbalanced sampling outcome.

### Reproduce
![1](figures/marie_1.png)
![1](figures/marie_2.png)
on SMAC
```bash
python train.py \
  --n_workers 1 \
  --env starcraft \
  --env_name ${map_name} \
  --seed 1 \
  --steps 100000 \
  --mode online \
  --tokenizer vq --decay 0.8 \
  --temperature 1.0 --sample_temp inf \
  --ce_for_av
```

on SMAX (JAX-based StarCraft Multi-Agent Challenge)
```bash
# Available maps: 3m, 2s3z, 25m, 3s5z, 8m, 5m_vs_6m, 10m_vs_11m, 27m_vs_30m, 3s5z_vs_3s6z, 3s_vs_5z, 6h_vs_8z, smacv2_5_units, smacv2_10_units, smacv2_20_units
python train.py --n_workers 1 --env smax --env_name ${map_name} --seed 1 --steps 1500000 --mode disabled \
  --tokenizer vq --decay 0.8 --temperature 1.0 --sample_temp inf --ce_for_av
```
It is worth noting that:

(1) in *3s_vs_5z*, we set ```--temperature 2.0``` to introduce more exploration and sample more diverse trajectories for world model learning;

(2) in *2m_vs_1z*, we find that enabling distributional loss for reward predicition can stablize the learning of MARIE, and thus we additionally enable ```--ce_for_r``` during this scenario experiment.

![1](figures/marie_3.png)
on MAMujoco

```bash
python train.py --n_workers 1 --env mamujoco --env_name ${map_name} --seed ${seed} --agent_conf ${agent_conf} --steps 1000000 \
  --mode online --tokenizer vq --decay 0.8 --temperature 1.0 --sample_temp inf --ce_for_r
```

on Gfootball, although we have not reported corresponding results in the paper yet, in our early experiments, we recommend the readers to use ```--tokenizer fsq``` to run MARIE, which can robustly discretize the observation in the gfootball env.

### PKL Data Export

MARIE automatically saves evaluation results as PKL files for research analysis. The PKL files are saved continuously during training:

- **Frequency**: Every 500 environment steps (after each evaluation)
- **Location**: `{date}_results/{env}/{map_name}-{tokenizer}/run{run_id}/marie_{map_name}_seed{seed}.pkl`
- **Format**: Dictionary with keys `['steps', 'eval_win_rates', 'eval_returns']`

This continuous saving ensures no data loss if training is interrupted.




## Acknowledgement

This repo is built upon [MAMBA - Scalable Multi-Agent Model-Based Reinforcement Learning](https://github.com/jbr-ai-labs/mamba). We thank the authors for their great work.

## Citation

If you find our paper or this repository helpful in your research or project, please consider citing our works using the following BibTeX citation:

```
@article{zhang2024decentralized,
  title={Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models},
  author={Zhang, Yang and Bai, Chenjia and Zhao, Bin and Yan, Junchi and Li, Xiu and Li, Xuelong},
  journal={arXiv preprint arXiv:2406.15836},
  year={2024}
}
```
