import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Connect to PyBullet GUI
p.connect(p.GUI)

# Set search path for built-in assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Physics settings
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)

# Load ground plane
#plane_id = p.loadURDF("plane.urdf")


def create_box(name, position, size, color, orientation=[0, 0, 0, 1]):
    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2]
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2],
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation
    )

    print(f"{name} created with ID:", body_id)
    return body_id


def create_cylinder_between(name, start, end, radius, color):
    """
    Creates a static cylinder between two 3D points.
    Useful for wires and diagonal supports.
    """
    start = np.array(start)
    end = np.array(end)
    midpoint = (start + end) / 2
    direction = end - start
    length = np.linalg.norm(direction)

    # Cylinder is naturally aligned along z-axis, so rotate it to match direction
    z_axis = np.array([0, 0, 1])
    direction_norm = direction / length

    axis = np.cross(z_axis, direction_norm)
    angle = math.acos(np.dot(z_axis, direction_norm))

    if np.linalg.norm(axis) < 1e-6:
        orientation = [0, 0, 0, 1]
    else:
        axis = axis / np.linalg.norm(axis)
        orientation = p.getQuaternionFromAxisAngle(axis, angle)

    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=length
    )

    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=midpoint.tolist(),
        baseOrientation=orientation
    )

    print(f"{name} created with ID:", body_id)
    return body_id


# -----------------------------
# Transmission pole dimensions
# -----------------------------
tower_x = 3
tower_height = 3.2
pole_radius = 0.12
crossarm_length = 3.2
crossarm_thickness = 0.12
wire_length = 7.0
wire_radius = 0.025

# Main vertical pole
main_pole = create_cylinder_between(
    name="Main Pole",
    start=[tower_x, 0, 0],
    end=[tower_x, 0, tower_height],
    radius=pole_radius,
    color=[0.45, 0.35, 0.22, 1]  # brown/wood colour
)

# Main horizontal crossarm
crossarm = create_box(
    name="Main Crossarm",
    position=[tower_x, 0, tower_height],
    size=[0.15, crossarm_length, crossarm_thickness],
    color=[0.55, 0.38, 0.18, 1]
)

# Small extra top beam / wire support beam
top_support_beam = create_box(
    name="Top Support Beam",
    position=[tower_x, 0, tower_height + 0.18],
    size=[0.10, crossarm_length * 0.8, 0.08],
    color=[0.55, 0.38, 0.18, 1]
)

# Diagonal braces like the image
left_brace = create_cylinder_between(
    name="Left Diagonal Brace",
    start=[tower_x, 0, tower_height - 0.7],
    end=[tower_x, -crossarm_length / 2 + 0.3, tower_height],
    radius=0.035,
    color=[0.55, 0.38, 0.18, 1]
)

right_brace = create_cylinder_between(
    name="Right Diagonal Brace",
    start=[tower_x, 0, tower_height - 0.7],
    end=[tower_x, crossarm_length / 2 - 0.3, tower_height],
    radius=0.035,
    color=[0.55, 0.38, 0.18, 1]
)

# -----------------------------
# SECOND TRANSMISSION POLE
# -----------------------------
tower2_x = 10

# Main vertical pole
main_pole_2 = create_cylinder_between(
    name="Main Pole 2",
    start=[tower2_x, 0, 0],
    end=[tower2_x, 0, tower_height],
    radius=pole_radius,
    color=[0.45, 0.35, 0.22, 1]
)

# Main horizontal crossarm
crossarm_2 = create_box(
    name="Main Crossarm 2",
    position=[tower2_x, 0, tower_height],
    size=[0.15, crossarm_length, crossarm_thickness],
    color=[0.55, 0.38, 0.18, 1]
)

# Top support beam
top_support_beam_2 = create_box(
    name="Top Support Beam 2",
    position=[tower2_x, 0, tower_height + 0.18],
    size=[0.10, crossarm_length * 0.8, 0.08],
    color=[0.55, 0.38, 0.18, 1]
)

# Left brace
left_brace_2 = create_cylinder_between(
    name="Left Brace 2",
    start=[tower2_x, 0, tower_height - 0.7],
    end=[tower2_x, -crossarm_length / 2 + 0.3, tower_height],
    radius=0.035,
    color=[0.55, 0.38, 0.18, 1]
)

# Right brace
right_brace_2 = create_cylinder_between(
    name="Right Brace 2",
    start=[tower2_x, 0, tower_height - 0.7],
    end=[tower2_x, crossarm_length / 2 - 0.3, tower_height],
    radius=0.035,
    color=[0.55, 0.38, 0.18, 1]
)

# -----------------------------
# THIRD TRANSMISSION POLE
# -----------------------------
tower3_x = 17

main_pole_3 = create_cylinder_between(
    name="Main Pole 3",
    start=[tower3_x, 0, 0],
    end=[tower3_x, 0, tower_height],
    radius=pole_radius,
    color=[0.45, 0.35, 0.22, 1]
)

crossarm_3 = create_box(
    name="Main Crossarm 3",
    position=[tower3_x, 0, tower_height],
    size=[0.15, crossarm_length, crossarm_thickness],
    color=[0.55, 0.38, 0.18, 1]
)

# -----------------------------
# FOURTH TRANSMISSION POLE
# -----------------------------
tower4_x = -4

main_pole_4 = create_cylinder_between(
    name="Main Pole 4",
    start=[tower4_x, 0, 0],
    end=[tower4_x, 0, tower_height],
    radius=pole_radius,
    color=[0.45, 0.35, 0.22, 1]
)

crossarm_4 = create_box(
    name="Main Crossarm 4",
    position=[tower4_x, 0, tower_height],
    size=[0.15, crossarm_length, crossarm_thickness],
    color=[0.55, 0.38, 0.18, 1]
)

# -----------------------------
# WIRES BETWEEN THE TWO POLES
# -----------------------------
wire_z = tower_height + 0.05
wire_z2 = tower_height + 0.15

wire_1 = create_cylinder_between(
    name="Left Wire",
    start=[tower_x, -crossarm_length / 2, wire_z],
    end=[tower2_x, -crossarm_length / 2, wire_z],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

wire_2 = create_cylinder_between(
    name="Middle Wire",
    start=[tower_x, 0, wire_z2 + 0.08],
    end=[tower2_x, 0, wire_z2 + 0.08],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

wire_3 = create_cylinder_between(
    name="Right Wire",
    start=[tower_x, crossarm_length / 2, wire_z],
    end=[tower2_x, crossarm_length / 2, wire_z],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

# -----------------------------
# EXTRA WIRES
# -----------------------------

# Wires between tower 4 and tower 1
create_cylinder_between(
    name="Wire Left Extension",
    start=[tower4_x, -crossarm_length / 2, wire_z],
    end=[tower_x, -crossarm_length / 2, wire_z],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

create_cylinder_between(
    name="Wire Middle Extension",
    start=[tower4_x, 0, wire_z2 + 0.08],
    end=[tower_x, 0, wire_z2 + 0.08],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

create_cylinder_between(
    name="Wire Right Extension",
    start=[tower4_x, crossarm_length / 2, wire_z],
    end=[tower_x, crossarm_length / 2, wire_z],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

# Wires between tower 2 and tower 3
create_cylinder_between(
    name="Wire Left Extension 2",
    start=[tower2_x, -crossarm_length / 2, wire_z],
    end=[tower3_x, -crossarm_length / 2, wire_z],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

create_cylinder_between(
    name="Wire Middle Extension 2",
    start=[tower2_x, 0, wire_z2 + 0.08],
    end=[tower3_x, 0, wire_z2 + 0.08],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

create_cylinder_between(
    name="Wire Right Extension 2",
    start=[tower2_x, crossarm_length / 2, wire_z],
    end=[tower3_x, crossarm_length / 2, wire_z],
    radius=wire_radius,
    color=[0.02, 0.02, 0.02, 1]
)

# -----------------------------
# GREEN GRASS GROUND
# -----------------------------

grass = create_box(
    name="Grass Ground",
    position=[6.5, 0, 0.005],
    size=[30, 30, 0.02],
    color=[0.2, 0.7, 0.3, 1]
)

# -----------------------------
# TARGET ZONES
# -----------------------------

def create_sphere(name, position, radius, color):
    collision_shape = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=radius
    )

    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )

    print(f"{name} created with ID:", body_id)
    return body_id


# Point A: drone start position marker
point_a = create_sphere(
    name="Point A - Drone Start",
    position=[1.0, 0, 0.3],
    radius=0.18,
    color=[1, 0, 0, 0.8]  # red
)

# Point B: inspection/photo target near top of second tower
point_b = create_sphere(
    name="Point B - Inspection Target",
    position=[tower2_x, -0.9, 2.9],
    radius=0.25,
    color=[0, 1, 0, 0.7]  # green
)

# -----------------------------
# LABELS FOR START AND END
# -----------------------------

p.addUserDebugText(
    text="START",
    textPosition=[1.0, 0, 0.7],
    textColorRGB=[1, 0, 0],
    textSize=1.4
)

p.addUserDebugText(
    text="GOAL",
    textPosition=[tower2_x, -0.9, 3.3],
    textColorRGB=[0, 1, 0],
    textSize=1.4
)

# -----------------------------
# ENVIRONMENT OUTPUT FUNCTIONS
# -----------------------------

drone_start_pos = np.array([1.0, 0, 0.3])
target_pos = np.array([tower2_x, -0.9, 2.9])

# Placeholder drone position for now
# Later, replace this with the real drone position from p.getBasePositionAndOrientation(drone_id)
drone_pos = drone_start_pos.copy()
drone_velocity = np.array([0, 0, 0])
drone_orientation = np.array([0, 0, 0])


def get_distance_to_target(drone_pos, target_pos):
    return np.linalg.norm(target_pos - drone_pos)


def check_collision(drone_id=None):
    """
    Placeholder collision checker.
    Later, once a real drone is loaded, this will check contact points.
    """
    if drone_id is None:
        return False

    contacts = p.getContactPoints(bodyA=drone_id)
    return len(contacts) > 0


def get_observation():
    """
    Returns the information needed by the RL teammate.
    """
    distance_to_target = get_distance_to_target(drone_pos, target_pos)
    collision_status = check_collision()

    observation = {
        "drone_position": drone_pos,
        "drone_velocity": drone_velocity,
        "drone_orientation": drone_orientation,
        "target_position": target_pos,
        "distance_to_target": distance_to_target,
        "collision_status": collision_status
    }

    return observation


def reset_environment():
    """
    Resets the drone back to Point A.
    Later, this should reset the actual drone position.
    """
    global drone_pos, drone_velocity, drone_orientation

    drone_pos = drone_start_pos.copy()
    drone_velocity = np.array([0, 0, 0])
    drone_orientation = np.array([0, 0, 0])

    return get_observation()


def step_environment(action=None):
    """
    Steps the PyBullet simulation.
    Later, action will be used to move/control the drone.
    """
    p.stepSimulation()

    obs = get_observation()

    # Simple placeholder reward
    reward = -obs["distance_to_target"]

    # Episode ends if close to target or collision occurs
    done = obs["distance_to_target"] < 0.5 or obs["collision_status"]

    info = {
        "reached_target": obs["distance_to_target"] < 0.5,
        "collision": obs["collision_status"]
    }

    return obs, reward, done, info

# Set camera view
p.resetDebugVisualizerCamera(
    cameraDistance=7.5,
    cameraYaw=58,
    cameraPitch=-15,
    cameraTargetPosition=[6.5, 0, 2.5]
)

print("PyBullet environment running...")
print("Transmission pole with support beam and wires created.")
print("Close the window to stop the simulation.")

# Main simulation loop
obs = reset_environment()
print("Initial observation:", obs)

last_print_time = time.time()

while True:
    obs, reward, done, info = step_environment()

    if time.time() - last_print_time >= 3:
        print("Distance to target:", obs["distance_to_target"])
        print("Reward:", reward)
        print("Info:", info)
        last_print_time = time.time()

    time.sleep(1 / 240)
