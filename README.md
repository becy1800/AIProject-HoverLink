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

## Run training with cmd below (make sure you are in correct folder)
```bash
python train.py
```

```bash
tensorboard --logdir results/save-05.11.2026_11.30.14/tb
```

```bash
python test.py
```