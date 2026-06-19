"""
Minimal Differentiable Gaussian Splatting in PyTorch
---------------------------------------------------
This single-file reference implements a tiny, end-to-end differentiable
Gaussian splatting renderer and a toy training loop. It is intentionally
small and readable (not production-fast) so you can hack on it for your
master's project and benchmark ideas.

Key features
- 3D anisotropic Gaussians parameterized by center, rotation (as a 3-vector
  exponential-map), and log-scales.
- Perspective projection and covariance pushforward to the image plane.
- Depth sorting + alpha compositing in software (O(N * H * W)). Works for
  small N and small images; replace with tile-based splatting for speed.
- Fully differentiable in PyTorch (no custom CUDA), so you can prototype
  optimizers and losses.

This follows the spirit of Kerbl et al. 2023 ("3D Gaussian Splatting for
Real-Time Radiance Field Rendering"), but with many simplifications:
- RGB colors instead of spherical harmonics.
- No view-dependent appearance, no adaptive densification/splitting.
- Slow rasterizer (good for understanding, not for real-time use).

Usage
- Run this file directly: it will optimize ~512 Gaussians to match a single
  target image rendered from a known camera. Modify the TARGET generation to
  load real images/cameras (e.g., COLMAP) and add multi-view supervision.

Copyright: MIT License
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------ Utilities ------------------------------ #

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def se3_look_at(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Build a right-handed view (world-to-camera) matrix (4x4).
    eye, center, up: (..., 3)
    Returns: (..., 4, 4) world-to-camera matrix.
    """
    f = F.normalize(center - eye, dim=-1)
    s = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
    u = torch.cross(s, f, dim=-1)
    # Camera matrix (camera axes are rows in world space)
    R = torch.stack([s, u, -f], dim=-2)  # (..., 3, 3)
    t = -R @ eye[..., None]               # (..., 3, 1)
    W2C = torch.eye(4, device=eye.device).expand(eye.shape[:-1] + (4, 4)).clone()
    W2C[..., :3, :3] = R
    W2C[..., :3, 3:4] = t
    return W2C


def intrinsics(fx, fy, cx, cy):
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device())
    return K


def rodrigues(r):
    """Exponential map so(3)->SO(3) for rotation vectors r (...,3)."""
    theta = torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-12)
    k = r / theta
    K = torch.zeros(r.shape[:-1] + (3, 3), device=r.device)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=r.device)
    theta = theta[..., 0]
    R = I + torch.sin(theta)[..., None, None] * K + (1 - torch.cos(theta))[..., None, None] * (K @ K)
    return R


# --------------------------- Gaussian Model ---------------------------- #

@dataclass
class Camera:
    K: torch.Tensor        # (3,3)
    W2C: torch.Tensor      # (4,4)
    width: int
    height: int


class GaussianCloud(nn.Module):
    def __init__(self, n_gauss: int, init_bounds: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        lo, hi = init_bounds
        # Centers in world space
        self.mu = nn.Parameter(torch.empty(n_gauss, 3).uniform_(lo, hi))
        # Rotation as axis-angle (so(3))
        self.rotvec = nn.Parameter(torch.zeros(n_gauss, 3))
        # Log scales (sx, sy, sz) to keep positive
        self.log_scale = nn.Parameter(torch.zeros(n_gauss, 3))
        # Per-Gaussian RGB (0..1) and opacity (alpha in [0,1])
        self.rgb = nn.Parameter(torch.rand(n_gauss, 3))
        self.logit_opacity = nn.Parameter(torch.full((n_gauss, 1), -0.2))

    def get_cov3d(self) -> torch.Tensor:
        R = rodrigues(self.rotvec)
        S = torch.diag_embed(torch.exp(self.log_scale).clamp_min(1e-4))  # (N,3,3)
        Sigma = R @ (S @ S) @ R.transpose(-1, -2)  # (N,3,3)
        return Sigma

    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacity)  # (N,1)


# ------------------------- Projection Jacobian ------------------------- #

@torch.no_grad()
def _make_pixel_grid(h: int, w: int, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return x, y  # (H,W)


def project_gaussians(cam: Camera, mu_w: torch.Tensor, Sigma_w: torch.Tensor):
    """Project 3D Gaussians to screen-space means, covariances, and depths.

    Args:
        cam: Camera
        mu_w: (N,3) centers in world space
        Sigma_w: (N,3,3) covariances in world space
    Returns:
        mu_uv: (N,2) mean in pixel coordinates
        Sigma_uv: (N,2,2) covariance in pixel space
        z_cam: (N,) depth in camera space (positive in front)
    """
    K = cam.K  # (3,3)
    W2C = cam.W2C

    # Transform means to camera space
    mu_w_h = torch.cat([mu_w, torch.ones_like(mu_w[..., :1])], dim=-1)  # (N,4)
    mu_c = (W2C @ mu_w_h.T).T[..., :3]  # (N,3)
    z = mu_c[:, 2].clamp_min(1e-4)

    # d(pi)/dx Jacobian for perspective projection x_c -> u,v
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X, Y = mu_c[:, 0], mu_c[:, 1]
    invz = 1.0 / z
    # J: (N,2,3)
    J = torch.zeros(mu_c.shape[0], 2, 3, device=mu_c.device)
    J[:, 0, 0] = fx * invz
    J[:, 0, 2] = -fx * X * (invz ** 2)
    J[:, 1, 1] = fy * invz
    J[:, 1, 2] = -fy * Y * (invz ** 2)

    # Pushforward covariance: Sigma_uv = J * Sigma_c * J^T
    # (world covariance to camera is just rotation part of W2C)
    R_wc = W2C[:3, :3]
    Sigma_c = (R_wc @ Sigma_w @ R_wc.T)  # broadcast (N,3,3)
    Sigma_uv = J @ Sigma_c @ J.transpose(-1, -2)  # (N,2,2)

    # Means in pixels
    u = fx * X * invz + cx
    v = fy * Y * invz + cy
    mu_uv = torch.stack([u, v], dim=-1)

    return mu_uv, Sigma_uv, z


# ------------------------- Naive Software Raster ----------------------- #

def render_splats(mu_uv: torch.Tensor,
                  Sigma_uv: torch.Tensor,
                  z: torch.Tensor,
                  rgb: torch.Tensor,
                  opacity: torch.Tensor,
                  H: int, W: int,
                  bg: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
    """Brute-force differentiable rasterizer.

    Evaluates each Gaussian density on all pixels (O(N*H*W)), performs
    front-to-back alpha compositing using depth order.

    Returns: (H,W,3) float image in [0,1].
    """
    N = mu_uv.shape[0]
    device = mu_uv.device

    # Depth sort (front-to-back: small z first)
    order = torch.argsort(z)  # (N,)
    mu_uv = mu_uv[order]
    Sigma_uv = Sigma_uv[order]
    rgb = rgb[order]
    a = opacity[order, 0].clamp(0, 1)

    # --- Numerically stable inverse/determinant of 2x2 covariances --- #
    # Symmetrize and add a pixel-space floor so splats never collapse.
    # A 0.5 px std-dev floor avoids singularities early in training.
    Sigma_uv = 0.5 * (Sigma_uv + Sigma_uv.transpose(-1, -2))
    sigma_floor = 0.5  # pixels
    I2 = torch.eye(2, device=device)
    Sigma_uv = Sigma_uv + (sigma_floor ** 2) * I2[None]

    # Try Cholesky with escalating jitter; fall back to pinv for any stragglers.
    jitter = 1e-8
    max_tries = 5
    S = Sigma_uv
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(S)
        bad = info > 0
        if not bad.any():
            break
        S = S.clone()
        S[bad] = S[bad] + jitter * I2
        jitter *= 10.0
    if (info > 0).any():
        # Last-resort: pseudo-inverse (keeps autograd) for the remaining few
        inv = torch.linalg.pinv(S)
        det = torch.linalg.det(S).abs().clamp_min(1e-12)
    else:
        inv = torch.cholesky_inverse(L)
        diagL = torch.diagonal(L, dim1=-2, dim2=-1)  # (N,2)
        det = (diagL.prod(dim=-1) ** 2).clamp_min(1e-12)

    norm = 1.0 / (2.0 * math.pi * torch.sqrt(det))  # (N,)

    # Pixel grid
    x, y = _make_pixel_grid(H, W, device)
    x = x[None, ...].float()
    y = y[None, ...].float()

    # Accumulators
    C = torch.ones(3, device=device)
    C = C * torch.tensor(bg, device=device)
    out = torch.zeros(H, W, 3, device=device)
    T = torch.ones(H, W, device=device)  # transmittance

    for i in range(N):
        mu = mu_uv[i]  # (2,)
        S_inv = inv[i]
        g = norm[i]
        dx = torch.stack([x - mu[0], y - mu[1]], dim=-1)  # (1,H,W,2)
        # Mahalanobis distance and pdf value
        md2 = torch.einsum('...i,ij,...j->...', dx, S_inv, dx)  # (1,H,W)
        w = torch.exp(-0.5 * md2) * g  # (1,H,W)
        # Convert w to per-pixel alpha via opacity
        alpha = (a[i] * w).clamp(0.0, 1.0)[0]
        # Pre-multiplied color
        col = rgb[i][None, None, :]  # (1,1,3)
        out = out + (alpha[..., None] * T[..., None]) * col
        T = T * (1.0 - alpha)

    # Composite background
    out = out + T[..., None] * C
    return out


# ------------------------------ Training ------------------------------- #

class TinyGS(nn.Module):
    def __init__(self, n_gauss: int):
        super().__init__()
        self.cloud = GaussianCloud(n_gauss)

    def forward(self, cam: Camera, H: int, W: int):
        mu = self.cloud.mu
        Sigma = self.cloud.get_cov3d()
        rgb = self.cloud.rgb.sigmoid()  # keep in [0,1]
        opacity = self.cloud.opacity()
        mu_uv, Sigma_uv, z = project_gaussians(cam, mu, Sigma)
        img = render_splats(mu_uv, Sigma_uv, z, rgb, opacity, H, W)
        return img


def make_toy_target(H: int, W: int, device):
    """Create a synthetic target image: colored rectangles/circles."""
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    img = torch.ones(H, W, 3, device=device)
    # Blue circle
    r = ((xx - W*0.35)**2 + (yy - H*0.4)**2).sqrt()
    img[r < min(H, W)*0.18, 2] = 0.2
    img[r < min(H, W)*0.18, 0] = 0.2
    # Red square
    img[int(H*0.55):int(H*0.85), int(W*0.55):int(W*0.85), :] = torch.tensor([0.85, 0.2, 0.2], device=device)
    return img.clamp(0, 1)


def default_camera(H: int, W: int, fov_deg: float = 60.0):
    dev = device()
    fx = fy = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
    cx, cy = W / 2, H / 2
    K = intrinsics(fx, fy, cx, cy).to(dev)
    eye = torch.tensor([0.0, 0.0, 3.0], device=dev)
    center = torch.tensor([0.0, 0.0, 0.0], device=dev)
    up = torch.tensor([0.0, 1.0, 0.0], device=dev)
    W2C = se3_look_at(eye, center, up)
    return Camera(K=K, W2C=W2C, width=W, height=H)


def train_tinygs():
    torch.manual_seed(42)
    H, W = 96, 96
    cam = default_camera(H, W)
    target = make_toy_target(H, W, device())

    n_gauss = 256
    model = TinyGS(n_gauss).to(device())

    # Initialize centers roughly in front of the camera
    with torch.no_grad():
        model.cloud.mu[:, 2].uniform_(0.3, 1.0)        # keep all in front of cam
        model.cloud.mu[:, :2].uniform_(-0.6, 0.6)      # within view
        model.cloud.log_scale.data[:] = -0.3           # wider footprint (~0.7 world units)
        model.cloud.logit_opacity.data[:] = 0.5        # α ≈ 0.62


    opt = torch.optim.Adam(model.parameters(), lr=5e-2)

    for it in range(200):
        opt.zero_grad(set_to_none=True)
        pred = model(cam, H, W)
        # Simple L2 loss + mild opacity penalty to avoid over-accumulation
        loss_img = F.mse_loss(pred, target)
        loss_sparse = 1e-4 * model.cloud.opacity().mean()
        loss_size = 1e-4 * torch.exp(model.cloud.log_scale).mean()
        loss = loss_img + loss_sparse + loss_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (it + 1) % 50 == 0:
            psnr = -10.0 * torch.log10(loss_img).item()
            print(f"iter {it+1:04d}  loss={loss.item():.5f}  psnr={psnr:.2f}dB")
        if (it+1) % 25 == 0:
            with torch.no_grad():
                pred8 = (pred.clamp(0,1)*255).to(torch.uint8).cpu().numpy()
                import imageio.v3 as iio
                iio.imwrite(f'debug_pred_{it+1:04d}.png', pred8)

            # Visibility & footprint stats
            a = model.cloud.opacity().detach().cpu().numpy().squeeze()
            mu_uv, Sigma_uv, z = project_gaussians(cam, model.cloud.mu, model.cloud.get_cov3d())
            # 2×2 covariance -> average pixel stddev (approx)
            a11 = Sigma_uv[...,0,0]; a22 = Sigma_uv[...,1,1]
            pix_std = torch.sqrt((a11 + a22) * 0.5).detach().cpu().numpy()
            print(f"[{it+1}] mean α={a.mean():.3f}, med α={np.median(a):.3f}, "
                f"mean pixσ={pix_std.mean():.2f}, "
                f"infrustum={(z>0).float().mean().item():.2f}")

    # Save result
    pred = model(cam, H, W).detach().cpu()
    import imageio.v3 as iio
    iio.imwrite('prediction.png', (pred.numpy() * 255).astype('uint8'))
    tgt = target.detach().cpu()
    iio.imwrite('target.png', (tgt.numpy() * 255).astype('uint8'))
    print('Saved prediction.png and target.png')


if __name__ == '__main__':
    train_tinygs()
