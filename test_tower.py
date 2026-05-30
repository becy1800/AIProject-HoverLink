import os
import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from TowerAviary import TowerAviary

OBS = ObservationType('kin')
ACT = ActionType('pid')

# ── update this path after training ─────────────────────────
MODEL_PATH = "results_tower/save-05.12.2026_16.24.50/best_model.zip"


def test_policy():
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run train_tower.py first, then update MODEL_PATH.")
        return

    model = PPO.load(MODEL_PATH)
    env   = TowerAviary(gui=True, obs=OBS, act=ACT)
    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=1,
                    output_folder="results_tower/playback/")

    obs, _ = env.reset(seed=42, options={})
    start  = time.time()

    for i in range((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs2 = obs.squeeze()
        act2 = action.squeeze()
        logger.log(drone=0,
                   timestamp=i / env.CTRL_FREQ,
                   state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                   control=np.zeros(12))

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)

        if terminated:
            dist = info.get("dist_to_goal", "?")
            print(f"Episode ended at step {i} | dist_to_goal={dist:.3f} | reward={reward:.3f}")
            break
        if truncated:
            print(f"Episode truncated at step {i} | reward={reward:.3f}")
            break

    env.close()
    logger.plot()


if __name__ == "__main__":
    test_policy()
