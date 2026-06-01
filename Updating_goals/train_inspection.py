import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from drone_env import DroneInspectionEnv

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
# All wire endpoints across all 4 towers — drone will train on all of these
TOWER_GOALS = [
    [3.0,  -1.6, 3.25],  # tower1_left
    [3.0,   1.6, 3.25],  # tower1_right
    [10.0, -0.9, 2.9 ],  # tower2_left  (original trained target)
    [10.0,  1.6, 3.25],  # tower2_right
    [17.0, -1.6, 3.25],  # tower3_left
    [17.0,  1.6, 3.25],  # tower3_right
    [-4.0, -1.6, 3.25],  # tower4_left
    [-4.0,  1.6, 3.25],  # tower4_right
]

# Single fixed goal used for evaluation so we get a consistent benchmark
EVAL_TARGET_POS = [10.0, -0.9, 2.9]

LOG_DIR           = "./logs/"
MODEL_DIR         = "./models/"
BEST_MODEL_PATH   = os.path.join(MODEL_DIR, "best_model")
CHECKPOINT_PATH   = os.path.join(MODEL_DIR, "checkpoints")
VECNORM_PATH      = os.path.join(MODEL_DIR, "vec_normalize.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

N_ENVS          = 8
TOTAL_TIMESTEPS = 5_000_000
EVAL_FREQ       = 20_000
N_EVAL_EPISODES = 5


# -----------------------------------------------------------------------
# Custom callback: log success rate to console
# -----------------------------------------------------------------------
class SuccessRateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_successes = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reached_target" in info:
                self._episode_successes.append(int(info["reached_target"]))
        if len(self._episode_successes) >= 50:
            rate = np.mean(self._episode_successes[-50:])
            self.logger.record("rollout/success_rate_50ep", rate)
        return True


# -----------------------------------------------------------------------
# Build environments
# -----------------------------------------------------------------------
def make_env():
    # Pick a random goal each episode by passing a random one at env creation.
    # DroneInspectionEnv.reset() restores _known_goal each episode, so to get
    # a *different* goal each episode we subclass and override reset().
    target = TOWER_GOALS[np.random.randint(len(TOWER_GOALS))]
    env = DroneInspectionEnv(render_mode=None, max_episode_steps=1000,
                             use_perception=False, target_pos=target)

    # Wrap so that each episode picks a new random goal
    env = RandomGoalWrapper(env)
    env = Monitor(env)
    return env


def make_eval_env():
    # Evaluation always uses the original trained target for a fair benchmark
    env = DroneInspectionEnv(render_mode=None, max_episode_steps=1000,
                             use_perception=False, target_pos=EVAL_TARGET_POS)
    env = Monitor(env)
    return env


# -----------------------------------------------------------------------
# Wrapper: re-randomises the goal on every reset()
# -----------------------------------------------------------------------
import gymnasium as gym

class RandomGoalWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        new_goal = TOWER_GOALS[np.random.randint(len(TOWER_GOALS))]
        self.env._known_goal = np.array(new_goal, dtype=np.float32)
        self.env.target_pos  = np.array(new_goal, dtype=np.float32)
        return self.env.reset(**kwargs)


def main():
    print("Building parallel training environments with randomised goals...")
    train_env = make_vec_env(make_env, n_envs=N_ENVS)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = VecNormalize(
        make_vec_env(make_eval_env, n_envs=1),
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_PATH,
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    class CheckpointWithVecNorm(BaseCallback):
        def __init__(self, save_freq, save_path, vecnorm_env, verbose=1):
            super().__init__(verbose)
            self.save_freq = save_freq
            self.save_path = save_path
            self.vecnorm_env = vecnorm_env

        def _on_step(self):
            if self.n_calls % self.save_freq == 0:
                steps = self.num_timesteps
                self.model.save(os.path.join(self.save_path, f"drone_ppo_{steps}_steps"))
                self.vecnorm_env.save(os.path.join(self.save_path, f"vec_normalize_{steps}_steps.pkl"))
                if self.verbose:
                    print(f"Saved checkpoint + vecnorm at {steps:,} steps")
            return True

    checkpoint_callback = CheckpointWithVecNorm(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CHECKPOINT_PATH,
        vecnorm_env=train_env,
        verbose=1,
    )

    print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps with {N_ENVS} envs...")
    print("Monitor: tensorboard --logdir ./logs/\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback, SuccessRateCallback()],
    )

    model.save(os.path.join(MODEL_DIR, "drone_ppo_final"))
    train_env.save(VECNORM_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_DIR}")
    print(f"VecNormalize stats saved to {VECNORM_PATH}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()