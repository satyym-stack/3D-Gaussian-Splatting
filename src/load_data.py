import json
import numpy as np
import torch
from PIL import Image
import os

def load_synthetic_data(data_path, W, H):
    # Check if JSON file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"JSON file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    # 1. Calculate Intrinsics (K)
    fov_x = data['camera_angle_x']
    focal_length = W / (2 * np.tan(fov_x / 2))
    cx, cy = W / 2, H / 2

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    images = []
    poses = []

    # 2. Load Poses and Images
    scene_dir = os.path.dirname(data_path)  # Directory containing the JSON
    for frame in data['frames']:
        # Load image (and convert to PyTorch tensor on CPU)
        img_path = os.path.join(scene_dir, frame['file_path'] + '.png')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = np.array(Image.open(img_path)) / 255.0
        images.append(torch.tensor(img, dtype=torch.float32))  # Keep on CPU

        # Load and adjust Camera-to-World pose
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)

        # Coordinate System Adjustment (Blender to GS standard)
        # Rotate to match X right, Y up, Z back
        rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        # Pad rotation to 4x4 for proper matrix multiplication with homogeneous transform
        rot_4x4 = np.eye(4, dtype=np.float32)
        rot_4x4[:3, :3] = rot
        c2w = rot_4x4 @ c2w

        # Convert to World-to-Camera for rendering
        w2c = np.linalg.inv(c2w)

        poses.append(torch.tensor(w2c, dtype=torch.float32))  # Keep on CPU

    return images, poses, K, H, W