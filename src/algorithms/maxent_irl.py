import os
import sys
import signal
import csv
import random
import numpy as np
import ot

from joblib import Parallel, delayed
from hydra.core.hydra_config import HydraConfig
import wandb

from ..algorithms.base_algo import BaseAlgo
from ..util.feature_gen import (
    find_feature_expectations,
    generate_trajectories,
)


def signal_handler(sig, frame):
    print("Interrupt received, shutting down...")
    wandb.finish()
    sys.exit(0)


class ContMaxEntIRL(BaseAlgo):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def train(self):
        self.configure_experiment()

        ######################################################################
        # TRAIN EXPERT
        #####################################################################
        self.env = self.create_env()
        self.rollout_env = self.create_env()
        self.expert = self.create_agent(use_init_params=True)
        self.expert.policy.tensorboard_log = f"checkpoints/{self.exp_name}/logs/"
        if self.cfg["load_expert"]:
            self.expert.load(
                self.cfg["path_to_expert"],
                env=self.env,
                custom_objects={
                    "observation_space": self.env.observation_space,
                    "action_space": self.env.action_space,
                },
            )
            with open(f"checkpoints/{self.exp_name}/readme.txt", "w") as f:
                f.write(
                    f'This run uses the saved model from {self.cfg["path_to_expert"]}'
                )
        else:
            self.expert.learn(logname="Expert")
            self.expert.save(f"checkpoints/{self.exp_name}/files/ppo_expert")

        # TESTING
        expert_results = [0, 0]
        if self.cfg["testing"]:
            video_path = f"checkpoints/{self.exp_name}/files/ppo_expert"
            expert_results = self.expert.save_render(
                video_dir=video_path,
                test_num=self.cfg["test_num"],
                test_env=self.rollout_env,
            )
            print(f"Expert results: {expert_results}")

        # GENERATE DATA
        if self.cfg["load_data"]:
            expert_trajs, expert_ts = self.load_data(
                self.cfg["path_to_data"], self.cfg["n_trajs"]
            )
        else:
            expert_trajs, expert_ts, rs = self.collect_rollouts(self.expert.policy)
            self.save_data(expert_trajs, expert_ts, rs)
        print("Expert data size: ", expert_trajs.shape, expert_ts.shape)

        # global STATE_MINMAX
        state_minmax = np.vstack(
            [np.min(expert_trajs, axis=0), np.max(expert_trajs, axis=0)]
        )
        np.save(f"tmp/{self.cfg["env_name"]}_state_minmax.npy", state_minmax)

        if self.cfg["expert_only"]:
            return 0

        ######################################################################
        # TRAIN IRL
        #####################################################################
        print("Training IRL...")
        lr = self.cfg["lr"]
        epochs = self.cfg["epochs"]
        d_states = self.cfg["d_states"]
        use_adam = self.cfg["use_adam"]

        # reward model initialization
        self.alpha = np.array([random.uniform(-1, 1) for x in range(d_states)])
        alpha_path = f"checkpoints/{self.exp_name}/files/alpha_temp"
        np.save(alpha_path, self.alpha)

        feature_expectations_expert = find_feature_expectations(
            self.cfg, expert_trajs, expert_ts
        )

        # Adam optimizer parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 5e-8
        m_w = 0
        v_w = 0
        t = 0

        # IRL loop
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            # update env and agents
            self.cfg["wrapper_kwargs"]["reward_path"] = alpha_path
            self.env = self.create_env()
            self.agent = self.create_agent(use_init_params=True)
            self.agent.learn(logname="Agent")

            agent_trajs, agent_ts, _ = self.collect_rollouts(self.agent.policy)

            # compute gradient
            feature_expectations_learner = find_feature_expectations(
                self.cfg, agent_trajs, agent_ts
            )
            grad = feature_expectations_expert - feature_expectations_learner
            if self.cfg["grad_clip"]:
                grad = np.clip(grad, -1, 1)
            # print(agent_trajs.shape)

            # compute wasserstein distance for subset
            expert_size = min(20000, expert_trajs.shape[0])
            agent_size = min(20000, expert_trajs.shape[0])
            dmatrix = ot.dist(expert_trajs[:expert_size, :], agent_trajs[:agent_size, :])
            dist = ot.emd2([], [], dmatrix, numItermax=500000, numThreads=10)

            # log training results
            if self.cfg["track"]:
                self.log_data(epoch, self.alpha, grad, dist, lr)
                if self.cfg["testing"] and (
                    epoch % self.cfg["test_epoch"] == 0 or epoch == 1
                ):
                    video_path = f"checkpoints/{self.exp_name}/files/ppo_learned_reward_ep{epoch}"
                    agent_results = self.agent.save_render(
                        video_dir=video_path,
                        test_num=self.cfg["test_num"],
                        test_env=self.rollout_env,
                    )
                    self.run.log(
                        {"avg_test_reward/MEAN_reward_agent": agent_results[0]}, commit=False
                    )
                    self.run.log(
                        {"avg_test_reward/STD_reward_agent": agent_results[1]}, commit=False
                    )
                    self.run.log(
                        {"avg_test_reward/MEAN_reward_expert": expert_results[0]}, commit=False
                    )
                    self.run.log(
                        {"avg_test_reward/STD_reward_expert": expert_results[1]}, commit=False
                    )

            # save model and reward
            if epoch % self.cfg["save_freq"] == 0 or epoch == 1:
                pass
                self.agent.save(
                    f"checkpoints/{self.exp_name}/files/ppo_learned_reward_ep{epoch}"
                )
                np.save(
                    f"checkpoints/{self.exp_name}/files/alpha_ep" + str(epoch),
                    self.alpha,
                )
                np.save(
                    f"checkpoints/{self.exp_name}/files/agent_trajs_learned"
                    + str(epoch),
                    agent_trajs,
                )
                np.save(
                    f"checkpoints/{self.exp_name}/files/agent_ts_learned" + str(epoch),
                    agent_ts,
                )

            # update alpha
            if use_adam:
                t += 1
                m_w = beta1 * m_w + (1 - beta1) * grad
                v_w = beta2 * v_w + (1 - beta2) * grad**2

                # Bias correction
                m_w_corr = m_w / (1 - beta1**t)
                v_w_corr = v_w / (1 - beta2**t)

                # Update weights using Adam
                self.alpha += lr * m_w_corr / (np.sqrt(v_w_corr) + epsilon)
                lr *= self.cfg["alpha_decay"]
            else:
                self.alpha += lr * grad
                lr *= self.cfg["alpha_decay"]
            np.save(alpha_path, self.alpha)

        self.run.finish()
        return dist

    def log_data(self, epoch, alpha, grad, dist, lr):
        self.run.log({"ep_logged": epoch}, commit=True)
        feat_keys = [f"feat_{i}" for i in range(alpha.shape[0])]
        for ind, item in enumerate(list(alpha)):
            self.run.log({f"alpha/{feat_keys[ind]}": item}, commit=False)
        for ind, item in enumerate(list(grad)):
            self.run.log({f"grad/{feat_keys[ind]}": item}, commit=False)
        self.run.log({"dist/wasserstein_distance": dist}, commit=False)
        self.run.log({"lr/lr": lr}, commit=False)

    def configure_experiment(self):

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if self.cfg["fast"]:
            self.set_simple_params()

        # if use wandb
        track = self.cfg["track"]
        group_name = self.cfg["group_name"]
        env_name = self.cfg["env_name"]
        self.exp_name = self.cfg["exp_name"]
        if track:
            wandb_project_name = self.cfg["wandb_project_name"]
            wandb_entity = self.cfg["wandb_entity"]
            notes = self.cfg["notes"]
            self.run = wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                # reinit=False,
                group=group_name,
                sync_tensorboard=self.cfg["sync_tb"],
                save_code=True,
                # job_type='outer_loop',
                # dir=log_path,
                # name=f"{exp_n}",
                notes=f"{notes}",
                config=self.cfg,
                settings=dict(init_timeout=120),
            )
            self.exp_name = f"{env_name}___group_{group_name}/run_{wandb.run.name}"
            self.cfg["exp_name"] = self.exp_name
            self.cfg["tensorboard_log"] = f"checkpoints/{self.exp_name}/logs/"
        self.cfg["hydra_config"] = HydraConfig.get()
        self.cfg["results_path"] = HydraConfig.get().sweep.dir
        if not os.path.exists(f"checkpoints/{self.exp_name}/files/"):
            os.makedirs(f"checkpoints/{self.exp_name}/files/")
        print(
            f'\n\n ---- Started ... |{self.exp_name} | \
                method - {self.cfg["feats_method"]} | \
                expert_only - {self.cfg["expert_only"]}\n'
        )

    def save_experiment_params(self, objective, expert_res, agent_res):

        params = []
        with open(f"checkpoints/{self.exp_name}/experiment_params.csv", "a") as fd:
            write = csv.writer(fd)
            params.append(str(objective))
            params.append(str(expert_res[0]))
            params.append(str(expert_res[1]))
            params.append(str(agent_res[0]))
            params.append(str(agent_res[1]))
            params.append(str(self.cfg["lr"]))
            params.append(str(self.cfg["gamma_feat"]))
            params.append(str(self.cfg["n_trajs"]))
            params.append(str(self.cfg["len_traj"]))
            params.append(str(self.cfg["epochs"]))
            params.append(str(self.cfg["d_states"]))
            params.append(str(self.cfg["total_timesteps"]))
            params.append(str(self.cfg["learning_rate"]))
            params.append(str(self.cfg["pi_size"]))
            params.append(str(self.cfg["samples_per_state"]))
            params.append(str(self.cfg["gamma"]))
            write.writerows(
                [
                    [
                        "objective",
                        "expert_mean",
                        "expert_std",
                        "agent_mean",
                        "agent_std",
                        "lr",
                        "gamma_feat",
                        "ntrajs",
                        "len_traj",
                        "epochs",
                        "d_states",
                        "total_timesteps",
                        "learning_rate",
                        "pi_size",
                        "samples_per_state",
                        "gamma",
                    ]
                ]
            )
            write.writerows([params])

    def collect_rollouts(self, policy):
        samples = self.cfg["samples_per_state"]
        trajs_num = self.cfg["n_trajs"] * samples
        nenvs = [self.create_env().envs[0] for _ in range(trajs_num)]
        res = Parallel(n_jobs=self.cfg["n_threads"], prefer="threads")(
            delayed(generate_trajectories)(self.cfg, policy, nenvs[seed], seed)
            for seed in range(0, trajs_num)
        )
        trs = [tr for (tr, id, r) in res]
        ids = [id for (tr, id, r) in res]
        rs = [r for (tr, id, r) in res]

        trajs = np.concatenate(trs)
        ts = np.concatenate(ids)
        rs = np.concatenate(rs)

        return trajs, ts, rs

    def load_data(self, path_to_data, n_trajs):
        expert_trajs = np.load(f"{path_to_data}expert_trajs.npy")
        expert_ts = np.load(f"{path_to_data}expert_ts.npy")
        expert_trajs = expert_trajs[: n_trajs * self.cfg["len_traj"]]
        expert_ts = expert_ts[: n_trajs * self.cfg["len_traj"]]
        with open(f"checkpoints/{self.exp_name}/readme.txt", "w") as f:
            f.write(
                f'This uses data of N={self.cfg["n_trajs"]} trajs from {self.cfg["path_to_data"]}'
            )
        return expert_trajs, expert_ts

    def save_data(self, expert_trajs, expert_ts, rs):
        np.save(
            f'checkpoints/{self.exp_name}/files/expert_trajs_{self.cfg["n_trajs"]}',
            expert_trajs,
        )
        np.save(
            f'checkpoints/{self.exp_name}/files/expert_ts_{self.cfg["n_trajs"]}',
            expert_ts,
        )
        np.save(
            f'checkpoints/{self.exp_name}/files/expert_rs_{self.cfg["n_trajs"]}', rs
        )

    def set_simple_params(self):
        # self.cfg.device='cpu'
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
