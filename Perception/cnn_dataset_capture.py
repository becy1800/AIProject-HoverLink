import os
import cv2
import csv
import numpy as np 
import pybullet as p 
import pybullet_data
import sys

# Add the Environment folder to the python path so we can import drone_sim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Environment')))
import drone_sim as lenv

print("Setting up PyBullet and generating dataset...")
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
lenv.build_scene(0)

os.makedirs("clean_dataset/images", exist_ok=True)
os.makedirs("clean_dataset/verifications", exist_ok=True)

csv_file = open("clean_dataset/labels.csv", "w", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "tower_x_norm", "tower_y_norm", "tower_conf", "wire_x_norm", "wire_y_norm", "wire_conf"])

TOWER_GOALS = {
    "tower1_left":  [3.0,  -1.6, 3.25],
    "tower1_right": [3.0,   1.6, 3.25],
    "tower2_left":  [10.0, -0.9, 2.9],   # current default
    "tower2_right": [10.0,  1.6, 3.25],
    "tower4_left":  [-4.0, -1.6, 3.25],
    "tower4_right": [-4.0,  1.6, 3.25],
}

num_samples_per_goal = 100
sample_idx = 0

# projection from 3d to 2d
def project_3d_to_2d(target_3d, view_mat, proj_mat, width, height_img):
    p_3d = np.array([target_3d[0], target_3d[1], target_3d[2], 1.0])
    p_view = np.dot(view_mat, p_3d)
    p_clip = np.dot(proj_mat, p_view)
    
    if p_clip[3] != 0:
        p_ndc = p_clip / p_clip[3]
    else:
        p_ndc = np.zeros(4)
        
    # calculate normalized coordinates
    u = (p_ndc[0] + 1.0) / 2.0 * width
    v = (1.0 - p_ndc[1]) / 2.0 * height_img
    
    x_norm = u / width
    y_norm = v / height_img
    
    conf = 1.0
    if p_clip[3] < 0: # behind camera
        conf = 0.0
    elif p_ndc[0] < -1 or p_ndc[0] > 1 or p_ndc[1] < -1 or p_ndc[1] > 1: # outside FOV
        conf = 0.0
        
    return x_norm, y_norm, conf, u, v

# image settings
img_width = 128
img_height = 128
fov = 60.0
near_val = 0.1
far_val = 1000.0
tower_height = 3.2

for goal_name, goal_coords in TOWER_GOALS.items():
    print(f"Generating samples for {goal_name}...")
    
    tower_x = goal_coords[0]
    tower_apex = np.array([tower_x, 0, tower_height])
    wire_endpoint = np.array(goal_coords)
    
    for i in range(num_samples_per_goal):
        distance = np.random.uniform(4.0, 8.0)
        angle = np.random.uniform(0, 2 * np.pi)
        height_offset = np.random.uniform(tower_height / 2, tower_height + 2)
        
        cam_pos = np.array([
            tower_x + distance * np.cos(angle),
            distance * np.sin(angle),
            height_offset
        ])
        
        # we point the camera roughly midway between the apex and the goal
        target_center = (tower_apex + wire_endpoint) / 2.0
        
        # add random noise to camera target so the tower isn't perfectly centered
        noise_x = np.random.uniform(-0.5, 0.5)
        noise_y = np.random.uniform(-0.5, 0.5)
        noise_z = np.random.uniform(-0.5, 0.5)
        cam_target = target_center + np.array([noise_x, noise_y, noise_z])
        
        # PyBullet camera setup
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=1.0,
            nearVal=near_val,
            farVal=far_val
        )
        
        # capture image
        w, h, rgb, dep, seg = p.getCameraImage(
            width=img_width,
            height=img_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            shadow=1,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb_img = np.reshape(rgb, (h, w, 4))[:, :, :3]
        
        # convert tuple matrices to numpy arrays 
        view_mat_np = np.array(view_matrix).reshape(4, 4).T
        proj_mat_np = np.array(proj_matrix).reshape(4, 4).T
        
        tx, ty, tc, tu, tv = project_3d_to_2d(tower_apex, view_mat_np, proj_mat_np, img_width, img_height)
        wx, wy, wc, wu, wv = project_3d_to_2d(wire_endpoint, view_mat_np, proj_mat_np, img_width, img_height)
        
        # saving data to csv file
        filename = f"frame_{sample_idx:04d}.png"
        csv_writer.writerow([filename, tx, ty, tc, wx, wy, wc])
        
        bgr_img = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join("clean_dataset/images", filename), bgr_img)
        
        ver_img = bgr_img.copy()
        if tc == 1.0:
            cv2.circle(ver_img, (int(tu), int(tv)), 2, (0, 0, 255), -1) # red for apex
        if wc == 1.0:
            cv2.circle(ver_img, (int(wu), int(wv)), 2, (0, 255, 0), -1) # green for wire endpoint
        cv2.imwrite(os.path.join("clean_dataset/verifications", filename), ver_img)
        
        sample_idx += 1
        
        if sample_idx % 50 == 0:
            print(f"Generated {sample_idx} total samples...")

csv_file.close()
p.disconnect()
print("Dataset generation complete!")
