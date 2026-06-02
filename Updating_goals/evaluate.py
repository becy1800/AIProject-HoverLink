import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

# from drone_env import DroneInspectionEnv
# at the top of your evaluate.py in /updating_goals
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'updating_goals'))

from drone_env import DroneInspectionEnv

# TARGET_POS = [10.0, -0.9, 2.9]
TOWER_GOALS = {
    "tower1_left":  [3.0,  -1.6, 3.25],
    "tower1_right": [3.0,   1.6, 3.25],
    "tower2_left":  [10.0, -0.9, 2.9],   # current default
    "tower2_right": [10.0,  1.6, 3.25],
    "tower3_left":  [17.0, -1.6, 3.25],
    "tower3_right": [17.0,  1.6, 3.25],
    "tower4_left":  [-4.0, -1.6, 3.25],
    "tower4_right": [-4.0,  1.6, 3.25],
}

TARGET_POS = TOWER_GOALS["tower2_right"]  # default


def make_eval_env(use_perception=True):
    env = DroneInspectionEnv(render_mode="human", max_episode_steps=1000,
                             use_perception=use_perception,
                             target_pos=TARGET_POS)
    env = Monitor(env)
    return env


def evaluate(model_path: str, vecnorm_path: str, n_episodes: int = 5,
             use_perception: bool = True):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    env = DummyVecEnv([lambda: make_eval_env(use_perception=use_perception)])

    if vecnorm_path and __import__("os").path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    n_goals = len(env.get_attr('goal_sequence')[0])

    successes = []
    distances = []

    for ep in range(n_episodes):
        goal_idx = ep % n_goals
        env.env_method('set_goal', goal_idx)
        goal_pos = env.get_attr('target_pos')[0]
        print(f"\n--- Episode {ep+1}: Goal {goal_idx} @ {goal_pos} ---")

        obs = env.reset()
        ep_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            steps += 1

            if done[0]:
                info_dict = info[0]
                dist      = info_dict.get("distance_to_target", float("inf"))
                reached   = info_dict.get("reached_target", False)
                collision = info_dict.get("collision", False)
                oob       = info_dict.get("out_of_bounds", False)
                success   = False
                if reached:
                    success = True
                    reason = "REACHED GOAL"
                elif collision:
                    reason = "COLLISION"
                elif oob:
                    reason = "OUT OF BOUNDS (hit ceiling/ground)"
                else:
                    reason = "TIMEOUT"
                successes.append(int(success))
                distances.append(dist)
                print(f"  Result: {reason} | "
                      f"steps={steps} | final_dist={dist:.2f}m | reward={ep_reward:.2f}")

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Success rate:    {np.mean(successes)*100:.1f}%")
    print(f"  Mean final dist: {np.mean(distances):.2f}m")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/best_model/best_model")
    parser.add_argument("--vecnorm",  default="models/vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-perception", action="store_true")
    args = parser.parse_args()

    evaluate(args.model, args.vecnorm, args.episodes,
             use_perception=not args.no_perception)
