import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from drone_env import DroneInspectionEnv


def make_eval_env(use_perception=True):
    env = DroneInspectionEnv(render_mode="human", max_episode_steps=1500,
                             use_perception=use_perception)
    env = Monitor(env)
    return env


def evaluate(model_path: str, vecnorm_path: str, n_episodes: int = 5):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    env = DummyVecEnv([lambda: make_eval_env(use_perception=True)])

    if vecnorm_path and __import__("os").path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    successes = []
    distances = []

    for ep in range(n_episodes):
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
                success = info_dict.get("reached_target", False)
                dist    = info_dict.get("distance_to_target", float("inf"))
                successes.append(int(success))
                distances.append(dist)
                print(f"  Episode {ep+1}: {'SUCCESS' if success else 'failed'} | "
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
    args = parser.parse_args()

    evaluate(args.model, args.vecnorm, args.episodes)
