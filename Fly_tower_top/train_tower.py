import os
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from TowerAviary import TowerAviary

# ============================================================
# Configuration
# ============================================================
OBS = ObservationType('kin')
ACT = ActionType('pid')
OUTPUT_FOLDER = 'results_tower'
TARGET_REWARD = 900.
PREVIOUS_MODEL = ""   # e.g. "results_tower/save-05.12.2026_.../best_model.zip"

if __name__ == "__main__":
    filename = os.path.join(OUTPUT_FOLDER, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(filename, exist_ok=True)

    train_env = make_vec_env(
        TowerAviary,
        env_kwargs=dict(obs=OBS, act=ACT),
        n_envs=4,
        seed=0
    )
    eval_env = TowerAviary(obs=OBS, act=ACT)

    print('[INFO] Action space:     ', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    if PREVIOUS_MODEL and os.path.exists(PREVIOUS_MODEL):
        print(f"[INFO] Loading existing model from {PREVIOUS_MODEL}")
        model = PPO.load(PREVIOUS_MODEL,
                         env=train_env,
                         tensorboard_log=filename + '/tb/')
    else:
        print("[INFO] Starting fresh training.")
        model = PPO('MlpPolicy',
                    train_env,
                    verbose=1,
                    tensorboard_log=filename + '/tb/')

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=TARGET_REWARD, verbose=1)
    eval_cb = EvalCallback(eval_env,
                           callback_on_new_best=stop_cb,
                           verbose=1,
                           best_model_save_path=filename + '/',
                           log_path=filename + '/',
                           eval_freq=1000,
                           deterministic=True,
                           render=False)

    print("Starting training...")
    model.learn(total_timesteps=int(1e6),
                callback=eval_cb,
                log_interval=100,
                reset_num_timesteps=False)
    model.save(filename + '/final_model.zip')
    print(f"Done. Model saved to {filename}")
    print(f"TensorBoard: tensorboard --logdir {filename}/tb/")
