import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import math
import time
from gymnasium import spaces


class DroneInspectionEnv(gym.Env):
    """
    Drone navigates from Point A (ground) to Point B (top of tower),
    avoiding transmission poles and wires.

    Observation (12,):
        drone_pos (3), drone_vel (3), drone_orient_euler (3),
        vec_to_target (3)   <- normalised direction + magnitude encoded

    Action (4,):
        Continuous force deltas [fx, fy, fz, yaw_torque] in [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, max_episode_steps=1000):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # ---------- spaces ----------
        obs_high = np.array([
            30, 30, 30,   # position
            10, 10, 10,   # velocity
            np.pi, np.pi, np.pi,  # euler angles
            30, 30, 30,   # vector to target (un-normalised for range)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # [thrust_x, thrust_y, thrust_z, yaw_torque]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([ 1,  1,  1,  1], dtype=np.float32),
        )

        # ---------- physics constants ----------
        self.GRAVITY = 9.81
        self.MASS = 0.5           # kg
        self.MAX_THRUST = 15.0    # N per axis
        self.MAX_TORQUE = 2.0     # Nm
        self.DT = 1 / 240

        # ---------- goal & start ----------
        self.start_pos = np.array([1.0, 0.0, 0.3], dtype=np.float32)
        tower2_x = 10.0
        self.target_pos = np.array([tower2_x, -0.9, 2.9], dtype=np.float32)
        self.reach_radius = 0.5   # success if within this distance

        # ---------- PyBullet ----------
        self._physics_client = None
        self._drone_id = None
        self._build_client()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)

        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=7.5, cameraYaw=58,
                cameraPitch=-15, cameraTargetPosition=[6.5, 0, 2.5],
                physicsClientId=self._physics_client,
            )
        else:
            self._physics_client = p.connect(p.DIRECT)

        cid = self._physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -self.GRAVITY, physicsClientId=cid)
        p.setTimeStep(self.DT, physicsClientId=cid)

        self._build_scene()
        self._spawn_drone()

    def _create_box(self, pos, size, color, orient=(0, 0, 0, 1)):
        cid = self._physics_client
        col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[s / 2 for s in size],
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[s / 2 for s in size],
                                  rgbaColor=color, physicsClientId=cid)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=pos,
                                 baseOrientation=orient, physicsClientId=cid)

    def _create_cylinder_between(self, start, end, radius, color):
        cid = self._physics_client
        start, end = np.array(start), np.array(end)
        mid = (start + end) / 2
        direction = end - start
        length = np.linalg.norm(direction)
        z_axis = np.array([0, 0, 1])
        dn = direction / length
        axis = np.cross(z_axis, dn)
        angle = math.acos(np.clip(np.dot(z_axis, dn), -1, 1))
        if np.linalg.norm(axis) < 1e-6:
            orient = (0, 0, 0, 1)
        else:
            orient = p.getQuaternionFromAxisAngle((axis / np.linalg.norm(axis)).tolist(), angle)

        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length, physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length,
                                  rgbaColor=color, physicsClientId=cid)
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=mid.tolist(),
                                 baseOrientation=orient, physicsClientId=cid)

    def _build_scene(self):
        cid = self._physics_client
        self._obstacle_ids = []

        tower_height = 3.2
        pole_radius = 0.12
        crossarm_length = 3.2
        wire_radius = 0.025
        wire_z = tower_height + 0.05
        wire_z2 = tower_height + 0.23
        wood = [0.45, 0.35, 0.22, 1]
        black = [0.02, 0.02, 0.02, 1]

        for tx in [-4, 3, 10, 17]:
            self._obstacle_ids.append(
                self._create_cylinder_between([tx, 0, 0], [tx, 0, tower_height], pole_radius, wood))
            self._obstacle_ids.append(
                self._create_box([tx, 0, tower_height],
                                 [0.15, crossarm_length, 0.12], [0.55, 0.38, 0.18, 1]))

        for t1, t2 in [(-4, 3), (3, 10), (10, 17)]:
            for y_off in [-crossarm_length / 2, 0, crossarm_length / 2]:
                wz = wire_z2 if y_off == 0 else wire_z
                self._obstacle_ids.append(
                    self._create_cylinder_between(
                        [t1, y_off, wz], [t2, y_off, wz], wire_radius, black))

        # Ground
        self._create_box([6.5, 0, 0.005], [30, 30, 0.02], [0.2, 0.7, 0.3, 1])

        # Goal marker (visual only)
        p.createVisualShape(p.GEOM_SPHERE, radius=0.25,
                            rgbaColor=[0, 1, 0, 0.5], physicsClientId=cid)
        if self.render_mode == "human":
            p.addUserDebugText("GOAL", self.target_pos.tolist(),
                               textColorRGB=[0, 1, 0], textSize=1.4, physicsClientId=cid)

    def _spawn_drone(self):
        cid = self._physics_client
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.15, physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                  rgbaColor=[0.2, 0.4, 1.0, 1.0], physicsClientId=cid)
        self._drone_id = p.createMultiBody(
            baseMass=self.MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self.start_pos.tolist(),
            physicsClientId=cid,
        )
        p.changeDynamics(self._drone_id, -1,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=cid)

    def _get_obs(self):
        cid = self._physics_client
        pos, orient_q = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=cid)
        vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=cid)
        pos = np.array(pos, dtype=np.float32)
        vel = np.array(vel, dtype=np.float32)
        euler = np.array(p.getEulerFromQuaternion(orient_q), dtype=np.float32)
        vec_to_target = (self.target_pos - pos).astype(np.float32)
        return np.concatenate([pos, vel, euler, vec_to_target])

    def _check_collision(self):
        cid = self._physics_client
        for obs_id in self._obstacle_ids:
            pts = p.getContactPoints(bodyA=self._drone_id, bodyB=obs_id, physicsClientId=cid)
            if pts:
                return True
        return False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        cid = self._physics_client

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        jitter = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        jitter[2] = abs(jitter[2])
        start = self.start_pos + jitter

        p.resetBasePositionAndOrientation(
            self._drone_id, start.tolist(), [0, 0, 0, 1], physicsClientId=cid)
        p.resetBaseVelocity(self._drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=cid)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        cid = self._physics_client
        self._step_count += 1

        force = (action[:3] * self.MAX_THRUST).tolist()
        force[2] += self.MASS * self.GRAVITY  # gravity compensation

        torque = [0, 0, float(action[3] * self.MAX_TORQUE)]

        p.applyExternalForce(
            self._drone_id, -1, force, [0, 0, 0], p.LINK_FRAME, physicsClientId=cid)
        p.applyExternalTorque(
            self._drone_id, -1, torque, p.LINK_FRAME, physicsClientId=cid)

        p.stepSimulation(physicsClientId=cid)

        obs = self._get_obs()
        pos = obs[:3]
        dist = np.linalg.norm(self.target_pos - pos)

        # Rewards
        r_goal      = float(np.exp(-dist ** 2))
        r_distance  = -float(dist) * 0.01
        collided    = self._check_collision()
        r_collision = -5.0 if collided else 0.0
        roll, pitch = obs[6], obs[7]
        r_stability = -0.1 * (abs(roll) + abs(pitch))
        out_of_bounds = bool(pos[2] < 0.05 or np.any(np.abs(pos[:2]) > 25))
        r_bounds    = -5.0 if out_of_bounds else 0.0
        r_smooth    = -0.001 * float(np.sum(action ** 2))

        reward = r_goal + r_distance + r_collision + r_stability + r_bounds + r_smooth

        reached    = bool(dist < self.reach_radius)
        terminated = reached or collided or out_of_bounds
        truncated  = self._step_count >= self.max_episode_steps

        info = {
            "distance_to_target": float(dist),
            "reached_target": reached,
            "collision": collided,
            "out_of_bounds": out_of_bounds,
        }

        if self.render_mode == "human":
            time.sleep(self.DT)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None
