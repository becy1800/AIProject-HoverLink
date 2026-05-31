import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class TowerAviary(BaseRLAviary):
    """Single agent RL: fly from the ground to the top of a tower without hitting it.

    The tower is a static cylinder at the world origin. The drone starts 1 m away
    at ground level and must reach the goal position just above the tower top.
    """

    TOWER_POS    = np.array([0.0, 0.0])  # XY base centre
    TOWER_RADIUS = 0.15                   # metres – roughly 30 cm diameter
    TOWER_HEIGHT = 1.0                    # metres

    # Extra observation dims appended to the base KIN observation:
    #   [dx_goal, dy_goal, dz_goal,  dx_tower, dy_tower,  dist_surface]
    _EXTRA_OBS = 6

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.PID):

        # Goal: just above the tower top
        self.TARGET_POS    = np.array([self.TOWER_POS[0], self.TOWER_POS[1],
                                       self.TOWER_HEIGHT + 0.1])
        self.EPISODE_LEN_SEC = 15
        self.tower_id = None  # filled by _addObstacles(), called inside super().__init__

        if initial_xyzs is None:
            initial_xyzs = np.array([[1.0, 0.0, 0.1]])  # 1 m from tower, near ground

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    ################################################################################

    def _addObstacles(self):
        """Create the tower as a static cylinder in PyBullet."""
        col_id = p.createCollisionShape(p.GEOM_CYLINDER,
                                        radius=self.TOWER_RADIUS,
                                        height=self.TOWER_HEIGHT,
                                        physicsClientId=self.CLIENT)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER,
                                     radius=self.TOWER_RADIUS,
                                     length=self.TOWER_HEIGHT,
                                     rgbaColor=[0.55, 0.35, 0.15, 1.0],
                                     physicsClientId=self.CLIENT)
        # PyBullet cylinders are positioned at their geometric centre
        self.tower_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[self.TOWER_POS[0], self.TOWER_POS[1], self.TOWER_HEIGHT / 2],
            physicsClientId=self.CLIENT
        )

    ################################################################################

    def _observationSpace(self):
        """KIN observation extended with tower/goal proximity features."""
        if self.OBS_TYPE != ObservationType.KIN:
            return super()._observationSpace()

        base = super()._observationSpace()
        lo = np.hstack([base.low,  np.full((1, self._EXTRA_OBS), -np.inf)])
        hi = np.hstack([base.high, np.full((1, self._EXTRA_OBS),  np.inf)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns base KIN obs + [rel_goal(3), rel_tower_xy(2), dist_surface(1)]."""
        base_obs = super()._computeObs()

        if self.OBS_TYPE != ObservationType.KIN:
            return base_obs

        pos = self._getDroneStateVector(0)[0:3]

        rel_goal      = self.TARGET_POS - pos                       # 3 values
        rel_tower_xy  = self.TOWER_POS  - pos[0:2]                  # 2 values
        dist_surface  = np.linalg.norm(rel_tower_xy) - self.TOWER_RADIUS  # 1 value

        extra = np.array([rel_goal[0], rel_goal[1], rel_goal[2],
                          rel_tower_xy[0], rel_tower_xy[1],
                          dist_surface], dtype=np.float32)

        return np.hstack([base_obs, extra.reshape(1, -1)])

    ################################################################################

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos   = state[0:3]

        dist_to_goal = np.linalg.norm(self.TARGET_POS - pos)

        # Large bonus for actually reaching the goal — increased radius makes it easier to trigger
        if dist_to_goal < 0.15:
            return 1000.0

        # Graduated near-goal bonus — pulls the drone in during final approach
        near_goal_bonus = max(0.0, (0.3 - dist_to_goal) * 5.0) if dist_to_goal < 0.3 else 0.0

        dist_xy       = np.linalg.norm(self.TOWER_POS - pos[0:2])
        dist_surface  = dist_xy - self.TOWER_RADIUS

        # Proximity penalty fades as drone climbs toward tower top
        # so the final approach above the tower is not penalised
        in_tower_z_band = 0.0 < pos[2] < self.TOWER_HEIGHT
        height_factor = max(0.0, (self.TOWER_HEIGHT - pos[2]) / self.TOWER_HEIGHT)
        if in_tower_z_band and dist_surface < 0.2:
            proximity_penalty = (0.2 - max(0.0, dist_surface)) * 1.0 * height_factor
        else:
            proximity_penalty = 0.0

        # Collision penalty fades toward zero as drone reaches tower top
        collision_penalty = 0.0
        if self.tower_id is not None:
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                          bodyB=self.tower_id,
                                          physicsClientId=self.CLIENT)
            if len(contacts) > 0:
                collision_penalty = -5.0 * height_factor

        tilt_penalty = float(np.linalg.norm(state[7:9])) * 0.1

        # Reward climbing — higher altitude always scores better
        # Extra bonus for clearing the tower top, where the goal is reachable
        height_reward = pos[2] * 0.5
        if pos[2] > self.TOWER_HEIGHT:
            height_reward += 1.0

        return float(2.0 - dist_to_goal + height_reward + near_goal_bonus - proximity_penalty + collision_penalty - tilt_penalty)

    ################################################################################

    def _computeTerminated(self):
        """Success if within 5 cm of goal; failure if touching the tower."""
        state = self._getDroneStateVector(0)

        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.05:
            return True

        if self.tower_id is not None:
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                          bodyB=self.tower_id,
                                          physicsClientId=self.CLIENT)
            if len(contacts) > 0:
                return True

        return False

    ################################################################################

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 3.0 or abs(state[1]) > 3.0
                or state[2] > self.TOWER_HEIGHT + 2.0 or state[2] < 0.05
                or abs(state[7]) > 0.7 or abs(state[8]) > 0.7):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        state    = self._getDroneStateVector(0)
        dist_xy  = float(np.linalg.norm(self.TOWER_POS - state[0:2]))
        return {
            "dist_to_goal":    float(np.linalg.norm(self.TARGET_POS - state[0:3])),
            "dist_to_surface": dist_xy - self.TOWER_RADIUS,
        }
