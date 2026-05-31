import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Environment'))

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
from gymnasium import spaces

from drone_sim import build_scene


class DroneInspectionEnv(gym.Env):
    """
    Drone navigates from Point A (ground) to Point B (top of tower),
    avoiding transmission poles and wires.

    Observation (12,):
        drone_pos (3), drone_vel (3), drone_orient_euler (3),
        vec_to_target (3)

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
            30, 30, 30,
            10, 10, 10,
            np.pi, np.pi, np.pi,
            30, 30, 30,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1], dtype=np.float32),
            high=np.array([ 1,  1,  1,  1], dtype=np.float32),
        )

        # ---------- physics constants ----------
        self.GRAVITY    = 9.81
        self.MASS       = 0.5
        self.MAX_THRUST = 15.0
        self.MAX_TORQUE = 2.0
        self.DT         = 1 / 240

        # ---------- goal & start ----------
        self.start_pos  = np.array([1.0, 0.0, 0.3], dtype=np.float32)
        self.target_pos = np.array([10.0, -0.9, 2.9], dtype=np.float32)
        self.reach_radius = 0.5

        # ---------- PyBullet ----------
        self._physics_client = None
        self._drone_id = None
        self._build_client()

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

        self._obstacle_ids = build_scene(cid, self.render_mode)
        self._spawn_drone()

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
        vel, _ = p.getBaseVelocity(self._drone_id, physicsClientId=cid)
        pos    = np.array(pos, dtype=np.float32)
        vel    = np.array(vel, dtype=np.float32)
        euler  = np.array(p.getEulerFromQuaternion(orient_q), dtype=np.float32)
        vec_to_target = (self.target_pos - pos).astype(np.float32)
        return np.concatenate([pos, vel, euler, vec_to_target])

    def _check_collision(self):
        cid = self._physics_client
        for obs_id in self._obstacle_ids:
            if p.getContactPoints(bodyA=self._drone_id, bodyB=obs_id, physicsClientId=cid):
                return True
        return False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        cid = self._physics_client

        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        jitter[2] = abs(jitter[2])
        start = self.start_pos + jitter

        p.resetBasePositionAndOrientation(
            self._drone_id, start.tolist(), [0, 0, 0, 1], physicsClientId=cid)
        p.resetBaseVelocity(self._drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=cid)

        return self._get_obs(), {}

    def step(self, action):
        cid = self._physics_client
        self._step_count += 1

        force = (action[:3] * self.MAX_THRUST).tolist()
        force[2] += self.MASS * self.GRAVITY

        p.applyExternalForce(
            self._drone_id, -1, force, [0, 0, 0], p.LINK_FRAME, physicsClientId=cid)
        p.applyExternalTorque(
            self._drone_id, -1, [0, 0, float(action[3] * self.MAX_TORQUE)],
            p.LINK_FRAME, physicsClientId=cid)

        p.stepSimulation(physicsClientId=cid)

        obs  = self._get_obs()
        pos  = obs[:3]
        dist = np.linalg.norm(self.target_pos - pos)

        collided      = self._check_collision()
        out_of_bounds = bool(pos[2] < 0.05 or np.any(np.abs(pos[:2]) > 25))
        reached       = bool(dist < self.reach_radius)
        roll, pitch   = obs[6], obs[7]

        r_goal      = max(0.0, 1.0 - float(dist) / 15.0)
        r_distance  = -float(dist) * 0.01
        r_collision = -5.0 if collided else 0.0
        r_stability = -0.1 * (abs(roll) + abs(pitch))
        r_bounds    = -5.0 if out_of_bounds else 0.0
        r_smooth    = -0.001 * float(np.sum(action ** 2))
        r_reach     = 200.0 if reached else 0.0

        reward     = r_goal + r_distance + r_collision + r_stability + r_bounds + r_smooth + r_reach
        terminated = reached or collided or out_of_bounds
        truncated  = self._step_count >= self.max_episode_steps

        if self.render_mode == "human":
            time.sleep(self.DT)

        return obs, reward, terminated, truncated, {
            "distance_to_target": float(dist),
            "reached_target":     reached,
            "collision":          collided,
            "out_of_bounds":      out_of_bounds,
        }

    def render(self):
        pass

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None
