import os
import sys
import signal
import random
import numpy as np
import ot
import wandb

from joblib import Parallel, delayed
from hydra.core.hydra_config import HydraConfig
from stable_baselines3.common.env_util import make_vec_env

from ..algorithms.base_algo import BaseAlgo
from ..agents.sb3_ppo import PPOAgent
from ..agents.sb3_sac import SACAgent
from ..envs.wrappers import (
    BaseEnvWrapper,
    WalkerWrapper,
    HopperWrapper,
    AntWrapper,
    CheetahWrapper,
)
from ..util.feature_gen import (
    find_feature_expectations,
    generate_trajectories,
)


def signal_handler(sig, frame):
    print("Interrupt received, shutting down...")
    wandb.finish()
    sys.exit(0)


class ContMaxEntIRL(BaseAlgo):
    """
    Continuous MaxEnt IRL algorithm implementation.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg["wrapper_kwargs"]["configs"] = cfg
        self.env = None
        self.rollout_env = None
        self.expert = None
        self.agent = None
        self.alpha = None
        self.run = None
        self.exp_name = self.cfg["exp_name"]
        self.checkpoint_dir = None
        self.files_dir = None

    def train(self):
        """
        IRL training:
        1. Setup (W&B, signal handlers, etc.).
        2. Train or load expert policy.
        3. Generate or load expert demonstrations.
        4. If expert_only, then train expert policy
        5. Run the IRL training loop (update reward via alpha, train new agent)
        """
        self._configure_experiment()

        # Train or load expert
        self.env = self._create_env()
        self.rollout_env = self._create_env()
        self._train_or_load_expert()

        # Test expert
        expert_results = self._test_expert()  # returns [mean, std] if tested
        print(f"Expert results: {expert_results}")

        # Generate or load expert demonstrations
        expert_trajs, expert_ts = self._generate_expert_data(self.expert.policy)

        if self.cfg["expert_only"]:
            return 0

        # IRL loop
        dist = self._run_irl_loop(expert_trajs, expert_ts, expert_results)
        if self.run is not None:
            self.run.finish()

        return dist

    def log_data(self, epoch, alpha, grad, dist, lr):
        """
        Logs to WandB run (if active).
        """
        if not self.run:
            return

        self.run.log({"ep_logged": epoch}, commit=True)
        feat_keys = [f"feat_{i}" for i in range(alpha.shape[0])]

        for ind, item in enumerate(alpha):
            self.run.log({f"alpha/{feat_keys[ind]}": item}, commit=False)

        for ind, item in enumerate(grad):
            self.run.log({f"grad/{feat_keys[ind]}": item}, commit=False)

        self.run.log({"dist/wasserstein_distance": dist}, commit=False)
        self.run.log({"lr/lr": lr}, commit=False)

    def _configure_experiment(self):
        """
        Sets up signal handlers, W&B experiment tracking, Hydra config paths, etc.
        Also defines checkpoint/log directories.
        """
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # If "fast", adjust some hyperparameters for a quick run(debug)
        if self.cfg["fast"]:
            self._set_simple_params()

        # Setup W&B tracking
        if self.cfg["track"]:
            wandb_project_name = self.cfg["wandb_project_name"]
            wandb_entity = self.cfg["wandb_entity"]
            notes = self.cfg["notes"]
            group_name = self.cfg["group_name"]
            env_name = self.cfg["env_name"]

            self.run = wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                group=group_name,
                sync_tensorboard=self.cfg["sync_tb"],
                save_code=True,
                notes=notes,
                config=self.cfg,
                settings=dict(init_timeout=120),
            )
            # Update exp_name to reflect W&B run
            self.exp_name = f"{env_name}___group_{group_name}/run_{wandb.run.name}"
            self.cfg["exp_name"] = self.exp_name
            # For SB3's TensorBoard logs:
            self.cfg["tensorboard_log"] = f"checkpoints/{self.exp_name}/logs/"

        # Hydra config references
        self.cfg["hydra_config"] = HydraConfig.get()
        self.cfg["results_path"] = HydraConfig.get().sweep.dir

        # Define directories for checkpoints and files
        self.checkpoint_dir = f"checkpoints/{self.exp_name}"
        self.files_dir = os.path.join(self.checkpoint_dir, "files")

        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)

        print(
            f"\n\n---- Started ... |{self.exp_name} | "
            f"method - {self.cfg['feats_method']} | "
            f"expert_only - {self.cfg['expert_only']}\n"
        )

    def _train_or_load_expert(self):
        """
        Creates or loads the expert policy (PPO/SAC).
        """
        self.expert = self._create_agent(use_init_params=True)
        # Update tensorboard log to our checkpoint directory
        self.expert.policy.tensorboard_log = os.path.join(self.checkpoint_dir, "logs")

        if self.cfg["load_expert"]:
            self.expert.load(
                self.cfg["path_to_expert"],
                env=self.env,
                custom_objects={
                    "observation_space": self.env.observation_space,
                    "action_space": self.env.action_space,
                },
            )
            with open(os.path.join(self.files_dir, "readme.txt"), "w") as f:
                f.write(
                    f'This run uses the saved model from {self.cfg["path_to_expert"]}'
                )
        else:
            self.expert.learn(logname="Expert")
            self.expert.save(os.path.join(self.files_dir, "ppo_expert"))

    def _test_expert(self):
        """
        Test the expert policy by rendering rollouts.
        """
        if not self.cfg["testing"]:
            return [0, 0]

        video_path = os.path.join(self.files_dir, "ppo_expert")
        expert_results = self.expert.save_render(
            video_dir=video_path,
            test_num=self.cfg["test_num"],
            test_env=self.rollout_env,
        )
        return expert_results

    def _generate_expert_data(self, expert_policy):
        """
        Generate or load the expert demonstration trajectories.
        """
        if self.cfg["load_data"]:
            expert_trajs, expert_ts = self._load_data(
                self.cfg["path_to_data"], self.cfg["n_trajs"]
            )
        else:
            expert_trajs, expert_ts, rs = self._collect_rollouts(expert_policy)
            self._save_data(expert_trajs, expert_ts, rs)

        print("Expert data size: ", expert_trajs.shape, expert_ts.shape)
        return expert_trajs, expert_ts

    def _run_irl_loop(self, expert_trajs, expert_ts, expert_results):
        """
        Runs the main IRL loop:
        1. Initialize parameters for the reward model.
        2. For each epoch:
           - Update env with new reward model
           - Train a new agent
           - Collect rollouts
           - Compute gradient vs. expert feature expectations
           - Compute Wasserstein distance
           - Log results
           - Save checkpoints
           - Update alpha (Adam or vanilla)
        Returns the final Wasserstein distance.
        """
        print("Training IRL...")
        lr = self.cfg["lr"]
        epochs = self.cfg["epochs"]
        d_states = self.cfg["d_states"]
        use_adam = self.cfg["use_adam"]

        # Initialize reward parameters alpha
        alpha_path = os.path.join(self.files_dir, "alpha_temp")
        self.alpha = np.array([random.uniform(-1, 1) for _ in range(d_states)])
        np.save(alpha_path, self.alpha)

        # Pre-compute expert feature expectations
        feature_expectations_expert = find_feature_expectations(
            self.cfg, expert_trajs, expert_ts
        )

        # Adam optimizer parameters
        beta1, beta2 = 0.9, 0.999
        epsilon = 5e-8
        m_w, v_w, t = 0, 0, 0

        dist = 0
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            # Update env & agent with current alpha
            self.cfg["wrapper_kwargs"]["reward_path"] = alpha_path
            self.env = self._create_env()
            self.agent = self._create_agent(use_init_params=True)
            self.agent.learn(logname="Agent")

            # Collect rollouts with new agent
            agent_trajs, agent_ts, _ = self._collect_rollouts(self.agent.policy)

            # Compute gradient
            feature_expectations_learner = find_feature_expectations(
                self.cfg, agent_trajs, agent_ts
            )
            grad = feature_expectations_expert - feature_expectations_learner
            if self.cfg["grad_clip"]:
                grad = np.clip(grad, -1, 1)

            # Compute Wasserstein distance (just for tracking)
            expert_size = min(20000, expert_trajs.shape[0])
            agent_size = min(20000, agent_trajs.shape[0])
            dmatrix = ot.dist(
                expert_trajs[:expert_size, :], agent_trajs[:agent_size, :]
            )
            dist = ot.emd2([], [], dmatrix, numItermax=500000, numThreads=10)

            # Log training results
            if self.cfg["track"]:
                self.log_data(epoch, self.alpha, grad, dist, lr)
                # Optionally test agent & compare to expert
                if self.cfg["testing"] and (
                    epoch % self.cfg["test_epoch"] == 0 or epoch == 1
                ):
                    video_path = os.path.join(
                        self.files_dir, f"ppo_learned_reward_ep{epoch}"
                    )
                    agent_results = self.agent.save_render(
                        video_dir=video_path,
                        test_num=self.cfg["test_num"],
                        test_env=self.rollout_env,
                    )
                    # Log to wandb
                    self.run.log(
                        {"avg_test_reward/MEAN_reward_agent": agent_results[0]},
                        commit=False,
                    )
                    self.run.log(
                        {"avg_test_reward/STD_reward_agent": agent_results[1]},
                        commit=False,
                    )
                    self.run.log(
                        {"avg_test_reward/MEAN_reward_expert": expert_results[0]},
                        commit=False,
                    )
                    self.run.log(
                        {"avg_test_reward/STD_reward_expert": expert_results[1]},
                        commit=False,
                    )

            # Save model & reward
            if epoch % self.cfg["save_freq"] == 0 or epoch == 1:
                self.agent.save(
                    os.path.join(self.files_dir, f"ppo_learned_reward_ep{epoch}")
                )
                np.save(os.path.join(self.files_dir, f"alpha_ep{epoch}"), self.alpha)
                np.save(
                    os.path.join(self.files_dir, f"agent_trajs_learned{epoch}"),
                    agent_trajs,
                )
                np.save(
                    os.path.join(self.files_dir, f"agent_ts_learned{epoch}"),
                    agent_ts,
                )

            # Update alpha (Adam or vanilla)
            if use_adam:
                t += 1
                m_w = beta1 * m_w + (1 - beta1) * grad
                v_w = beta2 * v_w + (1 - beta2) * (grad**2)

                # Bias correction
                m_w_corr = m_w / (1 - beta1**t)
                v_w_corr = v_w / (1 - beta2**t)

                # Adam update
                self.alpha += lr * m_w_corr / (np.sqrt(v_w_corr) + epsilon)
                lr *= self.cfg["alpha_decay"]
            else:
                self.alpha += lr * grad
                lr *= self.cfg["alpha_decay"]

            # Always save the latest alpha to disk
            np.save(alpha_path, self.alpha)

        return dist

    def _create_env(self):
        """
        Create and return a vectorized environment.
        """
        return make_vec_env(
            env_id=self.cfg["env_name"],
            n_envs=self.cfg["n_envs"],
            wrapper_class=eval(self.cfg["wrapper_class"]),
            wrapper_kwargs=self.cfg["wrapper_kwargs"],
        )

    def _create_agent(self, use_init_params):
        """
        Instantiate an RL Agent (PPO or SAC) for this environment.
        """
        if self.cfg["agent_name"] == "sb_ppo":
            return PPOAgent(self.cfg, env=self.env, use_init_params=use_init_params)
        elif self.cfg["agent_name"] == "sb_sac":
            return SACAgent(self.cfg, env=self.env, use_init_params=use_init_params)
        else:
            raise NotImplementedError(f"Unknown agent_name: {self.cfg['agent_name']}")

    def _collect_rollouts(self, policy):
        """
        Collect rollouts in parallel using joblib.
        Returns (trajectories, timesteps, rewards).
        """
        samples = self.cfg["samples_per_state"]
        trajs_num = self.cfg["n_trajs"] * samples

        # Create separate single-env instances for each parallel job
        nenvs = [self._create_env().envs[0] for _ in range(trajs_num)]
        res = Parallel(n_jobs=self.cfg["n_threads"], prefer="threads")(
            delayed(generate_trajectories)(self.cfg, policy, nenvs[seed], seed)
            for seed in range(trajs_num)
        )

        trajs, ts, rs = [np.concatenate(group) for group in zip(*res)]
        import pdb; pdb.set_trace()
        return trajs, ts, rs

    def _load_data(self, path_to_data, n_trajs):
        """
        Load expert trajs/ts from disk and truncate to 'n_trajs * len_traj'.
        """
        expert_trajs = np.load(f"{path_to_data}expert_trajs.npy")
        expert_ts = np.load(f"{path_to_data}expert_ts.npy")

        max_size = n_trajs * self.cfg["len_traj"]
        expert_trajs = expert_trajs[:max_size]
        expert_ts = expert_ts[:max_size]

        # Create a small readme
        readme_path = os.path.join(self.files_dir, "readme.txt")
        with open(readme_path, "w") as f:
            f.write(
                f"This uses data of N={self.cfg['n_trajs']} trajs "
                f"from {self.cfg['path_to_data']}"
            )
        return expert_trajs, expert_ts

    def _save_data(self, expert_trajs, expert_ts, rs):
        """
        Save generated expert data (trajs, ts, rewards) to disk.
        """
        np.save(
            os.path.join(self.files_dir, f"expert_trajs_{self.cfg['n_trajs']}"),
            expert_trajs,
        )
        np.save(
            os.path.join(self.files_dir, f"expert_ts_{self.cfg['n_trajs']}"),
            expert_ts,
        )
        np.save(os.path.join(self.files_dir, f"expert_rs_{self.cfg['n_trajs']}"), rs)

    def _set_simple_params(self):
        self.cfg.total_timesteps = 5000
        self.cfg.init_total_timesteps = 5000
        self.cfg.samples_per_state = 1
        self.cfg.n_trajs = 20
        self.cfg.samples_per_state = 1
        self.cfg.epochs = 7
        self.cfg.save_freq = 3
        self.cfg.test_epoch = 3
        self.cfg.test_num = 2
        self.cfg.group_name = "fast_debug"
        self.cfg.track = True
        self.cfg.use_wandb = True
        self.testing = True
        self.cfg.load_expert = False
        self.cfg.load_data = False
