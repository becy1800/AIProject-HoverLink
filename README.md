# Hover Link - AI for Robotics - Final Project

Welcome to the final project. Here we will be programming a drone with RL and PPO to fly from Point A to Point B.


## Step 1: 
### Create a Virtual Environment

Go to directory /AIProject-HoverLink then:
For Linux / macOS:
```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

For Windows
```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```


### Install dependencies:
```bash
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -r requirements.txt

```

## HOVER:
### Go to /AIProject-HoverLink/Hover
#### To train run:
```bash
python train.py
```
#### To view tensorboard run + with updated most recent results:
```bash
tensorboard --logdir results/save-05.11.2026_11.30.14/tb
```
#### To test run:
```bash
python test.py
```


## Fly_tower_top:
### Go to /AIProject-HoverLink/Fly_tower_top
#### To train run:
```bash
python train_tower.py
```

#### To test run:
```bash
python test_tower.py
```

## Obstacle_avoidance:
### Go to /AIProject-HoverLink/Obstacle_avoidance
#### To train run:
```bash
python train_inspection.py
```

#### To test run:
```bash
python evaluate.py --model models/best_model/best_model --vecnorm models/vec_normalize.pkl
```


## Perception:
### Go to /AIProject-HoverLink/Perception
#### To train run:
```bash
XXXXXXX
```

#### To test run:
```bash
XXXXXXX
```

## Environment:
### Go to /AIProject-HoverLink/Environment
This folder contains the PyBullet environments used to simulate the drone inspection task. The environments model electrical transmission towers, power lines, inspection targets, and terrain used for training and testing autonomous drone navigation algorithms.
### Files
#### drone_sim.py
A basic PyBullet environment used to visualise the transmission tower inspection scenario. The environment contains:
- Four transmission towers connected by power lines
- Start and goal markers representing the inspection mission
- A grass terrain area
- Observation, reset, and step functions for future reinforcement learning integration
- Camera positioning for viewing the inspection environment

###To run:
```bash
python drone_sim.py
