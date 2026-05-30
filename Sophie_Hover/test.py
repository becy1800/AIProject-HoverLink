import os
import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger

OBS = ObservationType('kin')
# ACT = ActionType('one_d_rpm')
ACT = ActionType('pid')

# ── update this path to your actual results folder ──────────
MODEL_PATH = "results/save-05.11.2026_11.30.14/final_model.zip"

def test_policy():
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run train.py first, then update MODEL_PATH to the saved folder.")
        return

    model = PPO.load(MODEL_PATH)
    env   = HoverAviary(gui=True, obs=OBS, act=ACT)
    logger = Logger(
        logging_freq_hz=int(env.CTRL_FREQ),
        num_drones=1,
        output_folder="results/playback/"
    )

    obs, _ = env.reset(seed=42, options={})
    start  = time.time()

    for i in range((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()

        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
            control=np.zeros(12)
        )

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)

        if truncated:
            print(f"Episode ended at step {i}, reward: {reward:.3f}")
            break
        if terminated:
            print(f"Drone is hovering perfectly at step {i}!")

    env.close()
    logger.plot()

if __name__ == "__main__":
    test_policy()