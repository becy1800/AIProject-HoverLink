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

## Simulation_environment:
### Go to /AIProject-HoverLink/Simulation_environment
You will see the environments currently used to simulate the test towers.
XXXX
XXXX