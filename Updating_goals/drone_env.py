import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Perception'))

import time
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from drone_sim_copy import build_scene

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DroneInspectionEnv(gym.Env):
    """
    Drone navigates from Point A (ground) to Point B (top of tower),
    avoiding transmission poles and wires.

    Observation (12,):
        drone_pos (3), drone_vel (3), drone_orient_euler (3),
        vec_to_target (3)

    Action (4,):
        Continuous force deltas [fx, fy, fz, yaw_torque] in [-1, 1]

    use_perception: if True, runs Claudia's CNN at episode start to locate
                    the target instead of using the hardcoded position.
                    Set False during training for speed.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # camera parameters (must match Claudia's dataset capture settings)
    CAM_W    = 128
    CAM_H    = 128
    CAM_FOV  = 60.0
    CAM_NEAR = 0.1
    CAM_FAR  = 1000.0

    def __init__(self, render_mode=None, max_episode_steps=1000, use_perception=True,
                 target_pos=None):
        super().__init__()
        self.render_mode      = render_mode
        self.max_episode_steps = max_episode_steps
        self.use_perception   = use_perception
        self._step_count      = 0

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
        self.start_pos    = np.array([1.0, 0.0, 0.3], dtype=np.float32)
        self.target_pos   = np.array(target_pos, dtype=np.float32) if target_pos is not None else None
        self._known_goal  = np.array(target_pos, dtype=np.float32) if target_pos is not None else None
        self.reach_radius = 0.4
        self._prev_dist   = None
        self._perception_cooldown = 0 

        self.goal_sequence = [
            [-4.0, -1.6, 2.9],   # tower4_left
            [-4.0,  1.6, 2.9],   # tower4_right
            [ 3.0, -1.6, 2.9],   # tower1_left
            [10.0, -0.9, 2.9],   # tower2_left
            [10.0,  1.6, 2.9],   # tower2_right
        ]

        self.current_goal_index = 0

        self.target_pos = np.array(self.goal_sequence[self.current_goal_index], dtype=np.float32)
        self._known_goal = self.target_pos.copy()

        # ---------- perception model ----------
        self._perception_model = None
        self._cam_transform    = None
        self._device           = None
        if self.use_perception:
            self._load_perception_model()

        # ---------- PyBullet ----------
        self._physics_client = None
        self._drone_id       = None
        self._build_client()

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _load_perception_model(self):
        if not TORCH_AVAILABLE:
            print("[WARNING] PyTorch not available — using hardcoded target.")
            return

        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'Perception', 'drone_vision_model.pth')

        if not os.path.exists(model_path):
            print("[WARNING] drone_vision_model.pth not found — using hardcoded target.")
            return

        try:
            from train_cnn import TargetDetectionCNN
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TargetDetectionCNN()
            model.load_state_dict(torch.load(model_path, map_location=self._device))
            model.eval()
            self._perception_model = model.to(self._device)
            self._cam_transform    = transforms.ToTensor()
            print(f"[INFO] Perception model loaded on {self._device}.")
        except Exception as e:
            print(f"[WARNING] Could not load perception model ({e}) — using hardcoded target.")

    def _get_target_from_perception(self):
        if self._perception_model is None:
            return None

        cid = self._physics_client
        pos, orient_q = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=cid)
        pos = np.array(pos)

        cam_pos    = pos + np.array([0, 0, 0.1])
        cam_target = np.array([self.target_pos[0], self.target_pos[1], self.target_pos[2]])  # ← use actual goal

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=[0, 0, 1],
            physicsClientId=cid,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.CAM_FOV, aspect=1.0,
            nearVal=self.CAM_NEAR, farVal=self.CAM_FAR,
            physicsClientId=cid,
        )

        _, _, rgb, depth_buf, _ = p.getCameraImage(
            width=self.CAM_W, height=self.CAM_H,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            physicsClientId=cid,
        )

        rgb_img   = np.reshape(rgb,       (self.CAM_H, self.CAM_W, 4))[:, :, :3].astype(np.uint8)
        depth_img = np.reshape(depth_buf, (self.CAM_H, self.CAM_W))

        with torch.no_grad():
            tensor = self._cam_transform(rgb_img).unsqueeze(0).to(self._device)
            pred   = self._perception_model(tensor)[0].cpu().numpy()

        tx = float(pred[0])
        ty = float(pred[1])
        tc = float(torch.sigmoid(torch.tensor(pred[2])))

        if tc < 0.5:
            return None

        from perception_wrapper import deproject_2d_to_3d
        apex_pos = deproject_2d_to_3d(tx, ty, depth_img, view_matrix, proj_matrix)
        if apex_pos is None:
            return None

        apex_x, _, _ = apex_pos
        if not (-10 < apex_x < 25):
            return None

        # Use perceived tower x-position; keep known goal y and z which are fixed by design
        world_pos = np.array([apex_x, self._known_goal[1], self._known_goal[2]], dtype=np.float32)

        # Reject if perception estimate is implausibly far from the known goal
        if np.linalg.norm(world_pos - self._known_goal) > 2.0:
            return None

        return world_pos

    # ------------------------------------------------------------------
    # PyBullet setup
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

        self._obstacle_ids = build_scene(cid, self.render_mode)
        self._add_goal_marker()
        self._spawn_drone()

    def _add_goal_marker(self):
        cid = self._physics_client

        if self.target_pos is None:
            goal_pos = np.array([10.0, -0.9, 2.9], dtype=np.float32)
        else:
            goal_pos = self.target_pos

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[1, 0, 0, 0.8], physicsClientId=cid)

        #p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis, basePosition=goal_pos.tolist(), physicsClientId=cid)
        self._goal_marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis, basePosition=goal_pos.tolist(), physicsClientId=cid)

        if self.render_mode == "human":
            #p.addUserDebugText("GOAL", [goal_pos[0], goal_pos[1], goal_pos[2] + 0.4], textColorRGB=[1, 0, 0], textSize=1.4, physicsClientId=cid)
            self._goal_text_id = p.addUserDebugText("GOAL", [goal_pos[0], goal_pos[1], goal_pos[2] + 0.4], textColorRGB=[1, 0, 0], textSize=1.4, physicsClientId=cid)

    def set_goal(self, goal_index):
        cid = self._physics_client

        self.current_goal_index = goal_index
        self.target_pos = np.array(self.goal_sequence[self.current_goal_index], dtype=np.float32)
        self._known_goal = self.target_pos.copy()

        if hasattr(self, "_goal_marker_id"):
            p.resetBasePositionAndOrientation(self._goal_marker_id, self.target_pos.tolist(), [0, 0, 0, 1], physicsClientId=cid)

        if hasattr(self, "_goal_text_id"):
            p.removeUserDebugItem(self._goal_text_id, physicsClientId=cid)

        if self.render_mode == "human":
            self._goal_text_id = p.addUserDebugText("GOAL", [self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.4], textColorRGB=[1, 0, 0], textSize=1.4, physicsClientId=cid)
    
    def _spawn_drone(self):
        cid = self._physics_client
        try:
            import pkg_resources
            # Pulls the path where the pip package stores its custom assets
            asset_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')
            cf2x_urdf_path = os.path.join(asset_path, "cf2x.urdf")
            # cf2x_urdf_path = ""
        except Exception:
            cf2x_urdf_path = ""
        if cf2x_urdf_path and os.path.exists(cf2x_urdf_path):
            self._drone_id = p.loadURDF(
                cf2x_urdf_path,
                basePosition=self.start_pos.tolist(),
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=cid
            )
            print(f"[INFO] Successfully loaded Crazyflie 2.x from: {cf2x_urdf_path}")
            
            # Explicitly scale up the base mass to match your obstacle environment parameters
            p.changeDynamics(self._drone_id, -1, mass=self.MASS, physicsClientId=cid)
        else:    
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3, physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3,
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
        vel, _        = p.getBaseVelocity(self._drone_id, physicsClientId=cid)
        pos   = np.array(pos, dtype=np.float32)
        vel   = np.array(vel, dtype=np.float32)
        euler = np.array(p.getEulerFromQuaternion(orient_q), dtype=np.float32)
        target = self.target_pos if self.target_pos is not None else pos
        return np.concatenate([pos, vel, euler, (target - pos).astype(np.float32)])

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

        rng    = np.random.default_rng(seed)
        jitter = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        jitter[2] = abs(jitter[2])
        start = self.start_pos + jitter

        p.resetBasePositionAndOrientation(
            self._drone_id, start.tolist(), [0, 0, 0, 1], physicsClientId=cid)
        p.resetBaseVelocity(self._drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=cid)

        if self.target_pos is not None:
            self._prev_dist = float(np.linalg.norm(self.target_pos - start))

        return self._get_obs(), {}

    def step(self, action):
        cid = self._physics_client
        self._step_count += 1

        force    = (action[:3] * self.MAX_THRUST).tolist()
        force[2] += self.MASS * self.GRAVITY

        p.applyExternalForce(
            self._drone_id, -1, force, [0, 0, 0], p.LINK_FRAME, physicsClientId=cid)
        p.applyExternalTorque(
            self._drone_id, -1, [0, 0, float(action[3] * self.MAX_TORQUE)],
            p.LINK_FRAME, physicsClientId=cid)

        p.stepSimulation(physicsClientId=cid)

        obs  = self._get_obs()
        pos  = obs[:3]

        # # activate perception within 6m of tower to refine navigation target
        # if self.use_perception and np.linalg.norm(pos - np.array([10.0, 0.0, 2.0])) < 6.0:
        #     perceived = self._get_target_from_perception()
        #     if perceived is not None:
        #         self.target_pos = perceived

        if self.use_perception:
            tower_center = np.array([self.target_pos[0], 0.0, 2.0])
            dist_to_tower = np.linalg.norm(pos - tower_center)
            
            # trigger earlier (10m) and use cooldown so it fires every 10 steps
            if dist_to_tower < 5.0 and self._perception_cooldown <= 0:
                perceived = self._get_target_from_perception()
                if perceived is not None:
                    self.target_pos = perceived
                    self._perception_cooldown = 10  # fire every 10 steps
            
            self._perception_cooldown -= 1

        dist = np.linalg.norm(self.target_pos - pos)

        # success measured against known goal, not perception-estimated target
        goal = self._known_goal if self._known_goal is not None else self.target_pos
        dist_to_goal = np.linalg.norm(goal - pos)

        collided      = self._check_collision()
        out_of_bounds = bool(pos[2] < 0.05 or pos[2] > 3.75 or np.any(np.abs(pos[:2]) > 25))
        reached       = bool(dist_to_goal < self.reach_radius)
        roll, pitch   = obs[6], obs[7]

        r_goal      = max(0.0, 1.0 - float(dist) / 15.0)
        r_distance  = -float(dist) * 0.01
        r_collision = -5.0 if collided else 0.0
        r_stability = -0.1 * (abs(roll) + abs(pitch))
        r_bounds    = -5.0 if out_of_bounds else 0.0
        r_smooth    = -0.001 * float(np.sum(action ** 2))
        r_reach     = 600.0 if reached else 0.0
        r_progress  = (self._prev_dist - dist) * 2.0 if self._prev_dist is not None else 0.0
        r_time      = -0.15
        self._prev_dist = float(dist)

        reward     = r_goal + r_distance + r_collision + r_stability + r_bounds + r_smooth + r_reach + r_progress + r_time
        
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
