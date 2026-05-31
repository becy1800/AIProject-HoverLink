# Obstacle Avoidance — Rebecca

## What it does
Trains a drone to fly from a starting position on the ground to an inspection point near the top of a transmission tower, avoiding poles and wires along the way.

The drone learns this behaviour using **Proximal Policy Optimisation (PPO)** — a reinforcement learning algorithm that rewards the drone for getting closer to the goal and penalises it for crashing or flying too high.

## Environment
Built in PyBullet with 4 transmission towers, 9 wires, and a target inspection point at `[10, -0.9, 2.9]` (near the top of tower 2). The drone starts at ground level `[1, 0, 0.3]` and must navigate 9.7m to the goal without hitting any obstacles.

## How to train
```bash
cd Obstacle_avoidance
python train_inspection.py
```

## How to evaluate
```bash
python evaluate.py --model models/checkpoints/drone_ppo_XXXXXX_steps \
                   --vecnorm models/checkpoints/vec_normalize_XXXXXX_steps.pkl
```
Replace `XXXXXX` with the checkpoint step number you want to test.

## Perception integration
When the drone gets within 6m of the tower, Claudia's CNN activates and identifies the target location from the drone's camera. The deprojection converts the 2D image prediction to 3D world coordinates, refining the drone's navigation for the final approach.

There are two separate roles:
- **Navigation target** — updated by perception each step. Guides where the drone flies toward the tower.
- **Success check** — always fixed at the real goal `[10, -0.9, 2.9]`. The drone only registers success when it physically reaches this position, regardless of small errors in the perception estimate.

This means perception steers the drone to the right area, but the finish line never moves.

## Key files
| File | Purpose |
|---|---|
| `drone_env.py` | PyBullet simulation environment and reward function |
| `train_inspection.py` | PPO training script |
| `evaluate.py` | Visual evaluation with GUI |
| `../Environment/drone_sim.py` | Shared scene geometry (towers, wires) |
