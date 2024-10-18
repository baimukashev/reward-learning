!/bin/bash

# train expert for selected env
python run_experiments.py \
  env_agent_algo=cheetah_ppo_maxent \
  group_name=expert_train \
  notes='' \
  load_expert=False \
  load_data=False \
  render=False \
  sync_tb=True \
  expert_only=True \
  seed=10 \
  init_total_timesteps=1e6\
  n_traj=200
