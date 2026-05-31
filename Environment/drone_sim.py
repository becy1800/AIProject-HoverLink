import pybullet as p
import pybullet_data
import time
import math
import numpy as np


def create_box(position, size, color, orientation=[0, 0, 0, 1], physicsClientId=0):
    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2],
        physicsClientId=physicsClientId
    )
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2],
        rgbaColor=color,
        physicsClientId=physicsClientId
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physicsClientId
    )


def create_cylinder_between(start, end, radius, color, physicsClientId=0):
    start = np.array(start)
    end = np.array(end)
    midpoint = (start + end) / 2
    direction = end - start
    length = np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    direction_norm = direction / length
    axis = np.cross(z_axis, direction_norm)
    angle = math.acos(np.clip(np.dot(z_axis, direction_norm), -1, 1))
    if np.linalg.norm(axis) < 1e-6:
        orientation = [0, 0, 0, 1]
    else:
        axis = axis / np.linalg.norm(axis)
        orientation = p.getQuaternionFromAxisAngle(axis.tolist(), angle)
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=length,
        physicsClientId=physicsClientId
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=length,
        rgbaColor=color,
        physicsClientId=physicsClientId
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=midpoint.tolist(),
        baseOrientation=orientation,
        physicsClientId=physicsClientId
    )


def create_sphere(position, radius, color, physicsClientId=0):
    collision_shape = p.createCollisionShape(
        p.GEOM_SPHERE, radius=radius, physicsClientId=physicsClientId)
    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=physicsClientId)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        physicsClientId=physicsClientId
    )


def build_scene(physics_client, render_mode=None):
    """
    Build the transmission pole scene.
    Returns a list of obstacle body IDs for collision detection.
    """
    cid = physics_client
    obstacle_ids = []

    # ---------- dimensions ----------
    tower_height = 3.2
    pole_radius = 0.12
    crossarm_length = 3.2
    wire_radius = 0.025
    wire_z = tower_height + 0.05
    wire_z2 = tower_height + 0.15

    wood  = [0.45, 0.35, 0.22, 1]
    brown = [0.55, 0.38, 0.18, 1]
    black = [0.02, 0.02, 0.02, 1]

    tower_x  = 3
    tower2_x = 10
    tower3_x = 17
    tower4_x = -4

    # ---------- Tower 1 (x=3) ----------
    obstacle_ids.append(create_cylinder_between(
        [tower_x, 0, 0], [tower_x, 0, tower_height], pole_radius, wood, physicsClientId=cid))
    obstacle_ids.append(create_box(
        [tower_x, 0, tower_height], [0.15, crossarm_length, 0.12], brown, physicsClientId=cid))
    obstacle_ids.append(create_box(
        [tower_x, 0, tower_height + 0.18], [0.10, crossarm_length * 0.8, 0.08], brown, physicsClientId=cid))
    obstacle_ids.append(create_cylinder_between(
        [tower_x, 0, tower_height - 0.7], [tower_x, -crossarm_length / 2 + 0.3, tower_height],
        0.035, brown, physicsClientId=cid))
    obstacle_ids.append(create_cylinder_between(
        [tower_x, 0, tower_height - 0.7], [tower_x, crossarm_length / 2 - 0.3, tower_height],
        0.035, brown, physicsClientId=cid))

    # ---------- Tower 2 (x=10) ----------
    obstacle_ids.append(create_cylinder_between(
        [tower2_x, 0, 0], [tower2_x, 0, tower_height], pole_radius, wood, physicsClientId=cid))
    obstacle_ids.append(create_box(
        [tower2_x, 0, tower_height], [0.15, crossarm_length, 0.12], brown, physicsClientId=cid))
    obstacle_ids.append(create_box(
        [tower2_x, 0, tower_height + 0.18], [0.10, crossarm_length * 0.8, 0.08], brown, physicsClientId=cid))
    obstacle_ids.append(create_cylinder_between(
        [tower2_x, 0, tower_height - 0.7], [tower2_x, -crossarm_length / 2 + 0.3, tower_height],
        0.035, brown, physicsClientId=cid))
    obstacle_ids.append(create_cylinder_between(
        [tower2_x, 0, tower_height - 0.7], [tower2_x, crossarm_length / 2 - 0.3, tower_height],
        0.035, brown, physicsClientId=cid))

    # ---------- Tower 3 (x=17) ----------
    obstacle_ids.append(create_cylinder_between(
        [tower3_x, 0, 0], [tower3_x, 0, tower_height], pole_radius, wood, physicsClientId=cid))
    obstacle_ids.append(create_box(
        [tower3_x, 0, tower_height], [0.15, crossarm_length, 0.12], brown, physicsClientId=cid))

    # ---------- Tower 4 (x=-4) ----------
    obstacle_ids.append(create_cylinder_between(
        [tower4_x, 0, 0], [tower4_x, 0, tower_height], pole_radius, wood, physicsClientId=cid))
    obstacle_ids.append(create_box(
        [tower4_x, 0, tower_height], [0.15, crossarm_length, 0.12], brown, physicsClientId=cid))

    # ---------- Wires (towers 4-1, 1-2, 2-3) ----------
    for t1, t2 in [(tower4_x, tower_x), (tower_x, tower2_x), (tower2_x, tower3_x)]:
        obstacle_ids.append(create_cylinder_between(
            [t1, -crossarm_length / 2, wire_z], [t2, -crossarm_length / 2, wire_z],
            wire_radius, black, physicsClientId=cid))
        obstacle_ids.append(create_cylinder_between(
            [t1, 0, wire_z2 + 0.08], [t2, 0, wire_z2 + 0.08],
            wire_radius, black, physicsClientId=cid))
        obstacle_ids.append(create_cylinder_between(
            [t1, crossarm_length / 2, wire_z], [t2, crossarm_length / 2, wire_z],
            wire_radius, black, physicsClientId=cid))

    # ---------- Ground (visual only) ----------
    create_box([6.5, 0, 0.005], [30, 30, 0.02], [0.2, 0.7, 0.3, 1], physicsClientId=cid)

    # ---------- Goal marker (visual only) ----------
    p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0, 1, 0, 0.5], physicsClientId=cid)

    if render_mode == "human":
        p.addUserDebugText("GOAL",  [tower2_x, -0.9, 3.3],
                           textColorRGB=[0, 1, 0], textSize=1.4, physicsClientId=cid)
        p.addUserDebugText("START", [1.0, 0, 0.7],
                           textColorRGB=[1, 0, 0], textSize=1.4, physicsClientId=cid)

    return obstacle_ids


# -----------------------------------------------------------------------
# Standalone visualisation — run this file directly to preview the scene
# -----------------------------------------------------------------------
if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)

    build_scene(0, render_mode="human")

    # Point A: drone start position marker
    create_sphere([1.0, 0, 0.3], 0.18, [1, 0, 0, 0.8])
    # Point B: inspection target
    create_sphere([10, -0.9, 2.9], 0.25, [0, 1, 0, 0.7])

    p.resetDebugVisualizerCamera(
        cameraDistance=7.5, cameraYaw=58,
        cameraPitch=-15, cameraTargetPosition=[6.5, 0, 2.5]
    )

    print("PyBullet environment running. Close the window to stop.")

    drone_start_pos = np.array([1.0, 0, 0.3])
    target_pos      = np.array([10, -0.9, 2.9])
    drone_pos       = drone_start_pos.copy()
    drone_velocity  = np.array([0, 0, 0])
    drone_orientation = np.array([0, 0, 0])

    def get_observation():
        dist = np.linalg.norm(target_pos - drone_pos)
        return {
            "drone_position":     drone_pos,
            "drone_velocity":     drone_velocity,
            "drone_orientation":  drone_orientation,
            "target_position":    target_pos,
            "distance_to_target": dist,
            "collision_status":   False,
        }

    def reset_environment():
        global drone_pos, drone_velocity, drone_orientation
        drone_pos         = drone_start_pos.copy()
        drone_velocity    = np.array([0, 0, 0])
        drone_orientation = np.array([0, 0, 0])
        return get_observation()

    def step_environment(action=None):
        p.stepSimulation()
        obs    = get_observation()
        reward = -obs["distance_to_target"]
        done   = obs["distance_to_target"] < 0.5 or obs["collision_status"]
        info   = {"reached_target": obs["distance_to_target"] < 0.5,
                  "collision": obs["collision_status"]}
        return obs, reward, done, info

    obs = reset_environment()
    print("Initial observation:", obs)

    last_print = time.time()
    while True:
        obs, reward, done, info = step_environment()
        if time.time() - last_print >= 3:
            print("Distance to target:", obs["distance_to_target"])
            last_print = time.time()
        time.sleep(1 / 240)
