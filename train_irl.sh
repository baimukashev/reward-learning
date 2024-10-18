#!/bin/bash

python run_experiments.py \
  env_agent_algo=cheetah_ppo_maxent \
  group_name=group_name \
  feats_method=proposed \
  d_states=12 \
  seed=400

python run_experiments.py \
  env_agent_algo=walker_ppo_maxent \
  group_name=group_name \
  feats_method=proposed \
  d_states=16 \
  seed=400

python run_experiments.py\
    env_agent_algo=ant_ppo_maxent\
    group_name=group_name\
    feats_method=proposed\
    d_states=20\
    seed=400

python run_experiments.py \
  env_agent_algo=hopper_ppo_maxent \
  group_name=group_name \
  feats_method=proposed \
  d_states=10 \
  seed=400