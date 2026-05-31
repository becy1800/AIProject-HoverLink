import gymnasium as gym
import torch
import torchvision.transforms as transforms
import numpy as np
import pybullet as p
from train_cnn import TargetDetectionCNN

# uses math in reverse to decode the 2D CNN predictions in reverse to decode the CNN's live guesses
# and replace the ground-truth target coordinates with the predicted 3D coordinates
def deproject_2d_to_3d(x_norm, y_norm, depth_img, view_matrix, proj_matrix):
    """Transforms 2D normalized CNN predictions back into 3D PyBullet world coordinates."""
    height, width = depth_img.shape
    pixel_x = np.clip(int(x_norm * width), 0, width - 1)
    pixel_y = np.clip(int(y_norm * height), 0, height - 1)
    
    depth_val = depth_img[pixel_y, pixel_x]
    if depth_val >= 1.0: return None # Hit the sky
    
    ndc_point = np.array([(x_norm * 2.0) - 1.0, 1.0 - (y_norm * 2.0), (depth_val * 2.0) - 1.0, 1.0])
    
    view_mat = np.array(view_matrix).reshape(4, 4).T
    proj_mat = np.array(proj_matrix).reshape(4, 4).T
    
    inv_view_proj = np.linalg.inv(proj_mat @ view_mat)
    world_pos_homo = inv_view_proj @ ndc_point
    
    return world_pos_homo[:3] / world_pos_homo[3]

class CNNPerceptionWrapper(gym.ObservationWrapper):
    """
    Wraps a PyBullet Drone Gym Environment. 
    Intercepts the RGB image, runs the CNN, performs deprojection, 
    and replaces the ground-truth target coordinates with the predicted 3D coordinates.
    """
    def __init__(self, env, model_path="drone_vision_model.pth"):
        super().__init__(env)
        
        # Load the CNN inside the Wrapper's initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = TargetDetectionCNN().to(self.device)
        self.cnn.load_state_dict(torch.load(model_path, map_location=self.device))
        self.cnn.eval() # Set to inference mode
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Adjust camera settings to match your training dataset
        self.img_width = 128
        self.img_height = 128
        self.fov = 60.0
        self.near_val = 0.1
        self.far_val = 1000.0

    def observation(self, obs):
        # This function runs automatically every time env.step() is called!
        
        # Get the exact position and orientation of the drone from PyBullet
        # (Assuming base environment has a way to get the drone's ID, e.g., self.env.unwrapped.drone_id)
        # If using gym-pybullet-drones, can get the state like this:
        try:
            drone_pos, drone_quat = p.getBasePositionAndOrientation(self.env.unwrapped.DRONE_IDS[0])
        except AttributeError:
            # Fallback if drone ID is not standard
            drone_pos = obs.get("drone_position", [1.0, 0, 0.3])
            drone_quat = p.getQuaternionFromEuler([0, 0, 0])
        
        # Compute Camera Matrices (Assuming camera points forward from the drone)
        rot_matrix = p.getMatrixFromQuaternion(drone_quat)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        forward_vector = rot_matrix.dot(np.array([1, 0, 0])) # Camera points along X-axis
        up_vector = rot_matrix.dot(np.array([0, 0, 1]))
        
        cam_target = np.array(drone_pos) + forward_vector
        
        view_matrix = p.computeViewMatrix(cameraEyePosition=drone_pos, cameraTargetPosition=cam_target, cameraUpVector=up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(fov=self.fov, aspect=1.0, nearVal=self.near_val, farVal=self.far_val)
        
        # Capture Live Image & Depth
        w, h, rgb, dep, seg = p.getCameraImage(
            width=self.img_width, height=self.img_height, 
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb_img = np.reshape(rgb, (h, w, 4))[:, :, :3]
        depth_img = np.reshape(dep, (h, w))
        
        # running the CNN because the CNN has to look at the brand new live image for every frame when the drone moves
        # and use the trained CNN model to predict the 2D coordinates of the tower and the wire endpoint 
        image_tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.cnn(image_tensor)[0]
            pred_tx, pred_ty, pred_tc = preds[0].item(), preds[1].item(), torch.sigmoid(preds[2]).item()
        
        # PPO needs to know where the tower is in 3D, not 2D, so we need to deproject the 2D coordinates to 3D
        tower_3d = None
        if pred_tc > 0.5: # If CNN is confident it sees the tower
            tower_3d = deproject_2d_to_3d(pred_tx, pred_ty, depth_img, view_matrix, proj_matrix)
            
        # Modify the observation fed to PPO!
        # Assuming obs is a dictionary, we replace the "true" target_position with our "predicted" one
        if isinstance(obs, dict) and tower_3d is not None:
            obs["target_position"] = tower_3d
            
        return obs
