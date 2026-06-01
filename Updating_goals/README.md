# Multiple Tower Nav - Sophie

## What it does
Trains a drone to fly from a starting position on the ground to an inspection point near the top of a transmission tower, to many other points.

The drone learns this behaviour using **Proximal Policy Optimisation (PPO)** — a reinforcement learning algorithm that rewards the drone for getting closer to the goal and penalises it for crashing or flying too high.

## Environment
Built in PyBullet with 4 transmission towers, 9 wires, and a target inspection point at `[10, -0.9, 2.9]` (near the top of tower 2). The drone starts at ground level `[1, 0, 0.3]` and must navigate to the left most tower, second left most, and second right most tower.

## How to train
```bash
cd Obstacle_avoidance_towers
python train_inspection.py
```

## How to evaluate
```bash
python evaluate.py --model models/checkpoints/drone_ppo_XXXXXX_steps \
                   --vecnorm models/checkpoints/vec_normalize_XXXXXX_steps.pkl
```
Replace `XXXXXX` with the checkpoint step number you want to test.

**Best converged model (run this for the demo):**
```bash
python evaluate.py --model models/checkpoints/drone_ppo_2200000_steps --vecnorm models/checkpoints/vec_normalize_2200000_steps.pkl --no-perception
```

To evaluate with perception: (only Tower 2 so far due to data from perception)
```bash
python evaluate.py --model models/checkpoints/drone_ppo_2200000_steps --vecnorm models/checkpoints/vec_normalize_2200000_steps.pkl
```