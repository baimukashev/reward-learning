import torch
from sbx import SAC

# from stable_baselines3.sac import SAC
from ..agents.base_agent import Agent


class SACAgent(Agent):
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

        self.policy = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                net_arch=dict(qf=[pi_size, pi_size], pi=[vf_size, vf_size])
            ),
            batch_size=self.cfg["batch_size"],
            learning_rate=learning_rate,
            gamma=gamma,
            learning_starts=self.cfg["learning_starts"],
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
        self.policy = SAC.load(path, env=env, custom_objects=custom_objects)
        return self.policy
