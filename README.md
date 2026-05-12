# AI for Robotics- Quiz 2: Reinforcement Learning

Welcome to the final project. Here we will be programming a drone with RL and PPO to fly from Point A to Point B.


### Step 1: Create a Virtual Environment

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


### Then install dependencies:
```bash
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -r requirements.txt

```

# Run training with cmd below (make sure you are in correct folder)
```bash
python train.py
```

Make sure you are in the /drone_rl folder:
```bash
tensorboard --logdir results/save-05.11.2026_11.30.14/tb
```

```bash
python test.py
```
