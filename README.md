# Hover Link - AI for Robotics - Final Project

Welcome to the final project. HoverLink is about creating an autonmous drone that will fly from point A, the ground, to point B, the top of a transmission tower. The purpose of HoverLink is to create the first step towards generating an autonomous mission in wire repair of transmission towers, are very dangerous human activity. There has been interaction with manual controlled drones to carry wires to the top of transmission towers, which we are inspired by and implement an AI take on. We use PPO policy to train the drone to fly from point A to point B using PID control of the movement, as well as CNN perception to analyse the apex location of a tower. A simulated transmission tower is created to support this environment.


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

Here, we use 
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
Trains a drone using PPO reinforcement learning to navigate from a ground-level start position to an inspection point near the top of a transmission tower, avoiding poles and wires. A perception CNN activates during the final approach to visually locate the target from the drone's camera.

### Go to /AIProject-HoverLink/Obstacle_avoidance
#### To train run:
```bash
python train_inspection.py
```

#### To test run (with perception):
```bash
python evaluate.py --model models/checkpoints/drone_ppo_600000_steps --vecnorm models/checkpoints/vec_normalize_600000_steps.pkl
```

#### To test run (obstacle avoidance only, no perception):
```bash
python evaluate.py --model models/checkpoints/drone_ppo_600000_steps --vecnorm models/checkpoints/vec_normalize_600000_steps.pkl --no-perception
```

#### To view training progress:
```bash
tensorboard --logdir logs/
```


## Perception:
This module trains a Convolutional Neural Network (CNN) to detect the tower apex and wire endpoint from the drone's RGB camera images. The CNN outputs a structured target estimate consisting of predicted (x, y) coordinates and visibility confidence scores, which serves as the observation state for the PPO navigation policy.

### Go to /AIProject-HoverLink/Perception
### Files
#### cnn_dataset_capture.py
Captures RGB images from the drone's camera and generates a labeled dataset containing the exact coordinates and visibility of the tower and wires.
#### To generate the training dataset:
```bash
python cnn_dataset_capture.py
```
#### train_cnn.py
Trains a Convolutional Neural Network (CNN) using the captured dataset to predict target coordinates.
#### To train the CNN model:
```bash
python train_cnn.py
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
### To run:
```bash
python drone_sim.py
