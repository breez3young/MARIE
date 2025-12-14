#!/bin/bash

### Available SMAX maps: 3m, 2s3z, 25m, 3s5z, 8m, 5m_vs_6m, 10m_vs_11m, 27m_vs_30m, 3s5z_vs_3s6z, 3s_vs_5z, 6h_vs_8z, smacv2_5_units, smacv2_10_units, smacv2_20_units

map_name="3s_vs_5z"
env="smax"
seed=1
cuda_device=0
mode=disabled
steps=10000

# --ce_for_av
CUDA_VISIBLE_DEVICES=${cuda_device} python train.py \
    --n_workers 1 --env ${env} --env_name ${map_name} \
    --seed ${seed} --steps $steps \
    --mode ${mode} \
    --tokenizer vq --decay 0.8 \
    --temperature 1.0 --sample_temp inf --ce_for_av
