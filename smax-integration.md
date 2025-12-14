# SMAX Integration for MARIE

This document describes the integration of **SMAX** (StarCraft Multi-Agent Challenge) environment from JaxMARL into MARIE.

## Overview

SMAX is a JAX-based implementation of the StarCraft Multi-Agent Challenge (SMAC) that provides fast, scalable multi-agent reinforcement learning scenarios. This integration brings SMAX support to MARIE with enhanced logging and continuous PKL data export for research analysis.

## Installation

### Prerequisites

Python 3.10+

### Install SMAX Dependencies

```bash
pip install "jax[cpu]==0.4.31"
pip install jaxlib
pip install jaxmarl
```

**Note**: SMAX uses CPU-only JAX to avoid CUDA initialization issues in Ray workers. This is automatically configured in the environment wrapper.

## Available SMAX Maps

MARIE supports all 14 SMAX scenarios from JaxMARL:

### Standard SMAC Maps
- **3m** - 3 Marines vs 3 Marines
- **8m** - 8 Marines vs 8 Marines  
- **25m** - 25 Marines vs 25 Marines
- **2s3z** - 2 Stalkers & 3 Zealots vs 2 Stalkers & 3 Zealots
- **3s5z** - 3 Stalkers & 5 Zealots vs 3 Stalkers & 5 Zealots

### Asymmetric Maps
- **5m_vs_6m** - 5 Marines vs 6 Marines
- **10m_vs_11m** - 10 Marines vs 11 Marines
- **27m_vs_30m** - 27 Marines vs 30 Marines
- **3s5z_vs_3s6z** - 3 Stalkers & 5 Zealots vs 3 Stalkers & 6 Zealots
- **3s_vs_5z** - 3 Stalkers vs 5 Zealots
- **6h_vs_8z** - 6 Hydralisks vs 8 Zealots

### SMACv2 Maps (Procedurally Generated)
- **smacv2_5_units** - 5v5 with randomized unit types and positions
- **smacv2_10_units** - 10v10 with randomized unit types and positions
- **smacv2_20_units** - 20v20 with randomized unit types and positions

## Usage

### Quick Start

Train MARIE on SMAX environment:

```bash
python train.py --env smax --env_name 3m --n_workers 4 --seed 1 --steps 1500000 --mode online --tokenizer vq --decay 0.8 --temperature 1.0 --sample_temp inf --ce_for_av
```

### Using the Training Script

A convenience script is provided at `train_smax.sh`:

```bash
#!/bin/bash

map_name="3m"
env="smax"
seed=1
steps=1500000
mode=online

CUDA_VISIBLE_DEVICES=0 python train.py \
    --env $env \
    --env_name $map_name \
    --n_workers 2 \
    --seed $seed \
    --steps $steps \
    --mode $mode \
    --tokenizer vq --decay 0.8 \
    --temperature 1.0 --sample_temp inf --ce_for_av
```

Run it:
```bash
chmod +x train_smax.sh
./train_smax.sh
```

### Command Line Arguments

| Argument | Description | Default | Example Values |
|----------|-------------|---------|----------------|
| `--env` | Environment type | `flatland` | `smax` |
| `--env_name` | SMAX map/scenario | `5_agents` | `3m`, `5m_vs_6m`, `3s5z` |
| `--n_workers` | Number of parallel workers | `2` | `2`, `4`, `8` |
| `--seed` | Random seed | `1` | `1`, `42`, `123` |
| `--steps` | Total training steps | `1000000` | `1500000`, `2000000` |
| `--mode` | Wandb logging mode | `disabled` | `online`, `offline`, `disabled` |

### Multi-Seed Experiments

Run multiple seeds for robust evaluation:

```bash
for seed in 1 2 3 4 5; do
    python train.py --env smax --env_name 3m --n_workers 4 --seed $seed --steps 1500000 --mode online
done
```

## Training Outputs

### Directory Structure

All training artifacts are saved to `{date}_results/smax/{map_name}-{tokenizer}/run{N}/`:

```
1214_results/smax/3m-vq/run1/
├── ckpt/
│   ├── model_200Ksteps.pth
│   ├── model_400Ksteps.pth
│   └── model_final.pth
├── agent/          # Backed up code
├── configs/        # Backed up configs
├── networks/       # Backed up network code
└── train.py        # Backed up training script
```

### PKL Data Export (Research Analysis) 📊

**Key Feature**: Training metrics are **continuously saved** to PKL files after each evaluation (every 500 steps), ensuring no data loss even if training is interrupted.

**Location**: `{date}_results/smax/marie_{map_name}_seed{seed}.pkl`

**Contents**:
```python
{
    'steps': [500, 1000, 1500, ...],           # Evaluation steps
    'eval_win_rates': [0.3, 0.5, 0.7, ...],    # Win rates over time
    'eval_returns': [50, 100, 150, ...]        # Cumulative returns
}
```

**Usage Example**:
```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('1214_results/smax/marie_3m_seed123.pkl', 'rb') as f:
    data = pickle.load(f)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(data['steps'], data['eval_win_rates'], label='Win Rate', marker='o')
plt.xlabel('Training Steps')
plt.ylabel('Win Rate')
plt.title('MARIE Learning Curve on 3m')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve_3m.png')
plt.show()

# Plot returns
plt.figure(figsize=(10, 6))
plt.plot(data['steps'], data['eval_returns'], label='Returns', marker='s', color='orange')
plt.xlabel('Training Steps')
plt.ylabel('Cumulative Returns')
plt.title('MARIE Returns on 3m')
plt.legend()
plt.grid(True)
plt.savefig('returns_3m.png')
plt.show()

# Statistical summary
print(f"Final win rate: {data['eval_win_rates'][-1]:.3f}")
print(f"Max win rate: {np.max(data['eval_win_rates']):.3f}")
print(f"Average returns: {np.mean(data['eval_returns']):.2f}")
```

### Wandb Integration

MARIE logs the following metrics to wandb:

- `win` - Episode win/loss (0 or 1)
- `returns` - Episode cumulative rewards
- `eval_win_rate` - Evaluation win rate (every 500 steps with 10 episodes)
- `eval_returns` - Evaluation cumulative returns
- `Agent/Returns`, `Agent/val_loss`, `Agent/actor_loss` - Training metrics

**Project name**: Uses the environment name (e.g., `smax`)

## Configuration

### Modify Hyperparameters

Edit SMAX-specific configs in `configs/dreamer/smax/`.

## Key Features

### ✅ Enhanced for MARIE
- **SMAX Environment Wrapper**: JAX-to-numpy conversion for PyTorch compatibility
- **SMAXLogWrapper**: Proper episode statistics tracking (battle_won, episode_return, episode_length)
- **CPU-Only JAX**: Avoids CUDA issues in Ray workers
- **Complete Config Suite**: Agent, Learner, Controller configs optimized for SMAX
- **Continuous PKL Export**: Data saved after every evaluation (500 steps) instead of only at training end
- **Multi-Environment Support**: SMAX coexists with existing MARIE environments (football, mamujoco, mpe)
- **Comprehensive Logging**: Win rate, returns, entropy tracking
- **Evaluation System**: 10-episode evaluation every 500 steps

## Environment Details

### SMAX Wrapper Features

The SMAX wrapper (`env/smax/SMAX.py`) provides:

1. **JAX to NumPy Conversion**: Converts JaxMARL's JAX arrays to numpy for PyTorch
2. **Episode Statistics**: Tracks battle_won, episode_return, win_rate via SMAXLogWrapper
3. **Available Actions**: Properly handles action masking for valid actions
4. **Observation Space**: Flattens observations to 1D arrays
5. **Discrete Actions**: 9 discrete actions per agent (move directions, attack, stop, no-op)

### Default Configuration

```python
{
    'scenario': map_name_to_scenario(env_name),
    'use_self_play_reward': False,
    'walls_cause_death': True,
    'see_enemy_actions': False,
    'action_type': 'discrete',
    'observation_type': 'unit_list',
}
```

## Troubleshooting

**Issue**: SMAX environment fails to initialize  
**Solution**: Ensure JAX dependencies are installed: `pip install jax[cpu]==0.4.31 jaxlib jaxmarl`

**Issue**: CUDA errors with SMAX  
**Solution**: SMAX uses CPU-only JAX by default (configured in `env/smax/SMAX.py`). Verify environment variables:
```python
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

**Issue**: PKL file not updating during training  
**Solution**: Check that evaluation is running (console should print eval stats every 500 steps). Verify `self.save_path` is correct.

**Issue**: Wandb login required  
**Solution**: Use `--mode disabled` to skip wandb, or run `wandb login` first

**Issue**: Out of memory during training  
**Solution**: Reduce `n_workers`, `BATCH_SIZE`, or `CAPACITY` in config files

## Comparison: MARIE vs mamba SMAX Integration

| Feature | MARIE |
|---------|-------|
| Evaluation Frequency | 500 steps | 
| PKL Save Behavior | Continuous (every eval) | 
| PKL Filename | `marie_{map}_seed{seed}.pkl` | 
| Wandb Project | Environment name (e.g., `smax`) | 
| Tokenizer Support | Yes (VQ, FSQ, etc.) | 
| Temperature Control | Yes | 

## References

- **JaxMARL**: [github.com/FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL)
- **SMAC**: [github.com/oxwhirl/smac](https://github.com/oxwhirl/smac)
- **MARIE**: Multi-Agent auto-Regressive Imagination for Efficient learning

## Support

For issues or questions:
1. Check this integration guide
2. Verify dependencies are correctly installed
3. Review console output for error messages
4. Check wandb logs for training metrics
5. Examine PKL files to ensure data is being saved

Happy training! 🚀
