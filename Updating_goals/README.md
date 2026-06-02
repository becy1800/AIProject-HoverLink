# Multiple Tower Nav - Sophie

## What it does
Trains a drone to fly from a starting position on the ground to inspection points near the top of transmission towers across 5 unique goal locations.

The drone learns this behaviour using **Proximal Policy Optimisation (PPO)** — a reinforcement learning algorithm that rewards the drone for getting closer to the goal and penalises it for crashing or flying too high.

## Environment
Built in PyBullet with 4 transmission towers, 9 wires, and 5 inspection goals positioned below the wire level at `z=2.9` across the towers. The drone starts at ground level `[1, 0, 0.3]` and navigates to each goal in sequence. A CNN-based perception system visually localises the target tower during the final approach.

## How to train
```bash
cd Updating_goals
python train_inspection.py
```

## How to evaluate

```bash
python evaluate.py --model models/checkpoints/drone_ppo_XXXXXX_steps \
                   --vecnorm models/checkpoints/vec_normalize_XXXXXX_steps.pkl
```
Replace `XXXXXX` with the checkpoint step number you want to test.

**Best converged model — no perception (run this for navigation demo):**
```bash
python evaluate.py --model models/checkpoints/drone_ppo_3000000_steps --vecnorm models/checkpoints/vec_normalize_3000000_steps.pkl --episodes 5 --no-perception
```

**Best converged model — with perception (run this for full pipeline demo):**
```bash
python evaluate.py --model models/checkpoints/drone_ppo_3000000_steps --vecnorm models/checkpoints/vec_normalize_3000000_steps.pkl --episodes 5
```

Perception is tested at the 3M step checkpoint and successfully reaches all 5 goal locations using the CNN to visually localise the target tower during the final approach.
