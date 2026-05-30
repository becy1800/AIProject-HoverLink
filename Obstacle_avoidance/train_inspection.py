import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from drone_env import DroneInspectionEnv

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
LOG_DIR       = "./logs/"
MODEL_DIR     = "./models/"
BEST_MODEL_PATH   = os.path.join(MODEL_DIR, "best_model")
CHECKPOINT_PATH   = os.path.join(MODEL_DIR, "checkpoints")
VECNORM_PATH      = os.path.join(MODEL_DIR, "vec_normalize.pkl")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

N_ENVS          = 8          # parallel environments
TOTAL_TIMESTEPS = 2_000_000
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
    env = DroneInspectionEnv(render_mode=None, max_episode_steps=1000)
    env = Monitor(env)
    return env


def main():
    print("Building parallel training environments...")
    train_env = make_vec_env(make_env, n_envs=N_ENVS)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = VecNormalize(
        make_vec_env(make_env, n_envs=1),
        norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
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

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CHECKPOINT_PATH,
        name_prefix="drone_ppo",
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
