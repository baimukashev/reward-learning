#!/bin/bash

# Test expert
python run_experiments.py \
  load_expert=True \
  testing=True \
  expert_only=True \
  load_data=False \
  render=True \
  seed=12 \
  use_wandb=False \
  track=False \
  env_agent_algo=cheetah_ppo_maxent \
  path_to_expert="checkpoints/HalfCheetah-v4___group_{}/run_{}/files/ppo_expert"
