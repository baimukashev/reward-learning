import torch
from sbx import PPO

# from stable_baselines3.ppo import PPO
from ..agents.base_agent import Agent


class PPOAgent(Agent):
    def __init__(self, cfg, env, use_init_params) -> None:
        super().__init__(cfg)
        self.configs = cfg
        self.env = env
        if use_init_params:
            size_key_prefix = "init_"
        else:
            size_key_prefix = ""
        pi_size = self.cfg[f"{size_key_prefix}pi_size"]
        vf_size = self.cfg.get(f"{size_key_prefix}vf_size", pi_size)
        gamma = self.cfg[f"{size_key_prefix}gamma"]
        learning_rate = self.cfg[f"{size_key_prefix}learning_rate"]
        self.total_timesteps = self.cfg[f"{size_key_prefix}total_timesteps"]

        self.policy = PPO(
            "MlpPolicy",
            env,
            normalize_advantage=True,
            policy_kwargs=dict(
                net_arch=dict(pi=[pi_size, pi_size], vf=[vf_size, vf_size])
            ),
            n_steps=self.cfg["n_steps"],
            batch_size=self.cfg["batch_size"],
            n_epochs=self.cfg["n_epochs"],
            learning_rate=learning_rate,
            gamma=gamma,
            use_sde=self.cfg["use_sde"],
            sde_sample_freq=self.cfg["sde_sample_freq"],
            verbose=self.cfg["verbose"],
            tensorboard_log=f"checkpoints/{self.cfg['exp_name']}/logs/",
            device=torch.device(
                self.cfg["device"],
            ),
        )

    def act(self, obs):
        action, *_ = self.policy.predict(torch.tensor(obs).float())
        return action

    def learn(self, logname=None):
        self.policy.learn(total_timesteps=self.total_timesteps, tb_log_name=logname)

    def save(self, path):
        self.policy.save(path)

    def load(self, path, env=None, custom_objects=None):
        self.policy = PPO.load(path, env=env, custom_objects=custom_objects)
        return self.policy
