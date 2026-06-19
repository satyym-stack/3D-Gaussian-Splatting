import numpy as np
import torch
import torch.nn as nn
import load_data as ld

class GaussianModel(torch.nn.Module):
    def __init__(self, initial_points):
        super().__init__()
        
        # 1. Position (Mean)
        self.means = nn.Parameter(initial_points[:, 0:3].requires_grad_(True)) 
        
        # 2. Opacity 
        self.opacities = nn.Parameter(initial_points[:, 3].requires_grad_(True)) # Logit scale
        
        # 3. Scaling (Log scale for positivity constraint)
        self.scaling = nn.Parameter(initial_points[:, 4:7].requires_grad_(True)) # log(s_x, s_y, s_z)
        
        # 4. Rotation (Quaternion)
        self.rotations = nn.Parameter(initial_points[:, 7:11].requires_grad_(True)) # Normalized quaternion
        
        # 5. Color (SH Coefficients - start with Degree 0 (constant color))
        # Initial color estimation (Degree 0 SH is just constant color)
        self.colors = nn.Parameter(initial_points[:, 11:14].requires_grad_(True)) 
        
        # Helper: Store the number of Gaussians
        self.n_gaussians = initial_points.shape[0]

if __name__ == '__main__':
    print("Starting")
    try:
        # Assuming NERF synthetic Lego dataset with 800x800 images
        W, H = 800, 800
        data_path = '../nerf_synthetic/lego/transforms_train.json'
        images, poses, K, H_loaded, W_loaded = ld.load_synthetic_data(data_path, W, H)

        # Example: Initialize Gaussians (replace with actual point cloud loading)
        # For a sample, generate random initial points (position, opacity, scales, rotations, colors)
        num_gaussians = 1000  # Example; adjust based on your point cloud
        initial_points = torch.randn(num_gaussians, 14)  # 3 pos + 1 opacity + 3 scales + 4 rotations + 3 colors

        model = GaussianModel(initial_points)
        print(f"Loaded {len(images)} images and {len(poses)} poses. Initialized model with {model.n_gaussians} Gaussians.")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}. Running with mock initialization for testing.")
        # Fallback: Mock data for testing (no images/poses loaded)
        W, H = 800, 800
        images, poses, K = [], [], np.eye(3, dtype=np.float32)  # Empty lists and identity K

        # Initialize Gaussians with random points
        num_gaussians = 100
        initial_points = torch.randn(num_gaussians, 14)
        model = GaussianModel(initial_points)
        print(f"Mock run: Initialized model with {model.n_gaussians} Gaussians (no data loaded).")
    except Exception as e:
        print(f"Error: {e}")