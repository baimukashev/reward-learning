import torch
import numpy as np
from .feature_select import select_feat_extractor


def generate_trajectories(cfg, policy, env, seed):
    trajs_obs = []
    trajs_steps = []
    trajs_rewards = []
    i = 0
    done = 0
    trunc = 0
    obs, _ = env.reset(seed=seed)
    while i < cfg["len_traj"] and not done and not trunc:
        action, *_ = policy.predict(torch.tensor(obs).float(), deterministic=False)
        next_obs, reward, done, trunc, info = env.step(action)
        trajs_obs.append(next_obs)
        trajs_steps.append(i)
        trajs_rewards.append(reward)
        obs = next_obs
        i += 1
    env.close()
    return np.array(trajs_obs), np.array(trajs_steps), np.array(trajs_rewards)


def find_feature_expectations(cfg, trajectories, steps):
    gamma = cfg["gamma_feat"]
    env_name = cfg["env_name"]
    feature_expectations = np.zeros(cfg["d_states"])
    for i, states in enumerate(trajectories):
        features = select_feat_extractor(env_name, states, cfg)  # phi(s)
        features_discounted = features * (gamma ** steps[i])  # phi(s) * (gamma ** time)
        feature_expectations += (
            features_discounted  # phi_exp += phi(s) * (gamma ** time)
        )

    feature_expectations /= trajectories.shape[0]
    return feature_expectations
