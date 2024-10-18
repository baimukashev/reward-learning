import numpy as np
import gymnasium as gym
from ..util.feature_gen import select_feat_extractor


class BaseEnvWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Base wrapper to handle observation storage and reward transformation.
    """

    def __init__(self, env: gym.Env, reward_path=None, scaler_path=None, configs=None):
        super().__init__(StoreObservation(env))
        if reward_path and reward_path != "None" and scaler_path and configs:
            alpha = np.load(reward_path + ".npy")
            self.env = TransformRewardLearnedCont(self.env, alpha, configs)
        elif reward_path is None:
            raise ValueError("reward path cannot be None if specified")


class TransformRewardLearnedCont(gym.RewardWrapper):
    """Transform the reward via an arbitrary function."""

    def __init__(self, env: gym.Env, alpha=None, configs=None):
        super().__init__(env)
        self.alpha = alpha
        self.configs = configs

    def reward(self, reward):
        state = self.get_wrapper_attr("temp_state")
        env_name = self.spec.id
        feature_expectations = select_feat_extractor(env_name, state, cfg=self.configs)
        reward = feature_expectations.dot(self.alpha)
        return reward


class StoreObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation):
        self.temp_state = observation
        return observation


class CropObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        """Resizes image observations to shape given by :attr:`shape`.
        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
        )
        super().__init__(env)

    def observation(self, observation):
        self.temp_state = observation[1:]
        return self.temp_state


class StoreAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        """Resizes image observations to shape given by :attr:`shape`.
        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        super().__init__(env)

    def action(self, action):
        self.temp_action = action
        return action


class NoEarlyTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        base_env = env.unwrapped  # Access the base environment without any wrappers
        if hasattr(base_env, "_terminate_when_unhealthy"):
            base_env._terminate_when_unhealthy = False
        else:
            print(
                "Warning: The environment does not support custom termination behavior."
            )


class WalkerWrapper(BaseEnvWrapper):
    """
    Specific wrapper for the CartPole environment.
    """

    def __init__(
        self, env, reward_path=None, env_name=None, scaler_path=None, configs=None
    ):
        env = NoEarlyTerminationWrapper(env)
        # env = NormalizeObservation(env)
        # env = NormalizeReward(env)
        super().__init__(env, reward_path, scaler_path, configs)


class CheetahWrapper(BaseEnvWrapper):
    """
    Specific wrapper for the CartPole environment.
    """

    def __init__(
        self, env, reward_path=None, env_name=None, scaler_path=None, configs=None
    ):
        super().__init__(env, reward_path, scaler_path, configs)


class AntWrapper(BaseEnvWrapper):
    """
    Specific wrapper for the CartPole environment.
    """

    def __init__(
        self, env, reward_path=None, env_name=None, scaler_path=None, configs=None
    ):
        env = NoEarlyTerminationWrapper(env)
        super().__init__(env, reward_path, scaler_path, configs)


class HopperWrapper(BaseEnvWrapper):
    """
    Specific wrapper for the CartPole environment.
    """

    def __init__(
        self, env, reward_path=None, env_name=None, scaler_path=None, configs=None
    ):
        env = NoEarlyTerminationWrapper(env)
        super().__init__(env, reward_path, scaler_path, configs)
