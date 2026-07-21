"""Shared observation preprocessing for the hinyeun glue DP3 policy.

Single source of truth for the camera-to-world transform, workspace crop, and
point-cloud sampling. The dataset converter and the deployment inference
server must both use these functions so train and deploy observations match.
"""
import numpy as np
import torch

# Unit vector pointing physically down in camera_color_optical_frame
# (dataset meta/orbbec_gravity.yaml, measured 2026-07-14).
GRAVITY_CAM = np.array([-0.033281993, 0.854071911, 0.519089086])

# 12-bit HEVC codes in the LeRobot dataset were encoded with depth_max=5.0 m;
# verified empirically against lerobot 0.6.0 decoding (ratio 5000/4096).
# Only relevant when decoding the packaged dataset videos, NOT live depth
# topics (which are already uint16 millimetres).
DEPTH_MM_PER_CODE = 5000.0 / 4095.0

# Workspace crop in the gravity-aligned frame: camera at origin, z up,
# x = camera optical axis projected onto the horizontal plane.
# Table planes sit at z ~ [-0.33, -0.27]; floor at ~ -1.31; back wall at x > 1.35.
WORK_SPACE = {
    'x': (0.20, 1.25),
    'y': (-0.90, 0.90),
    'z': (-0.45, 0.10),
}

NUM_POINTS = 1024
PIXEL_STRIDE = 2
PRE_SAMPLE = 16384


def gravity_rotation(g_down):
    """R with p_world = R @ p_cam: world z opposes gravity, world x is the
    camera forward axis projected onto the horizontal plane."""
    z_w = -g_down / np.linalg.norm(g_down)
    z_cam = np.array([0.0, 0.0, 1.0])
    x_w = z_cam - np.dot(z_cam, z_w) * z_w
    x_w /= np.linalg.norm(x_w)
    y_w = np.cross(z_w, x_w)
    return np.stack([x_w, y_w, z_w], axis=0)


R_G = gravity_rotation(GRAVITY_CAM)


def farthest_point_sampling_torch(points, num_points, device):
    """points: (N,3) float32 numpy. Returns (num_points,3) numpy."""
    pts = torch.from_numpy(points).to(device)
    n = pts.shape[0]
    if n <= num_points:
        pad = pts[torch.randint(0, n, (num_points - n,), device=device)]
        return torch.cat([pts, pad], dim=0).cpu().numpy()
    sel = torch.empty(num_points, dtype=torch.long, device=device)
    dist = torch.full((n,), float('inf'), device=device)
    sel[0] = 0
    last = pts[0]
    for i in range(1, num_points):
        d = torch.sum((pts - last) ** 2, dim=1)
        dist = torch.minimum(dist, d)
        idx = torch.argmax(dist)
        sel[i] = idx
        last = pts[idx]
    return pts[sel].cpu().numpy()


def depth_m_to_workspace_cloud(depth_m, K, rng,
                               stride=PIXEL_STRIDE,
                               pre_sample=PRE_SAMPLE,
                               num_points=NUM_POINTS,
                               device='cuda',
                               T_delta=None,
                               gravity=None,
                               work_space=None):
    """depth_m: (H,W) float32 metres, 0 = invalid. K: 3x3 intrinsics of the
    depth/color-registered frame. Returns (num_points, 3) float32 in the
    gravity-aligned frame.

    Migration overrides (all default to the original behaviour):
      T_delta   4x4 old-cam -> new-cam transform applied to the back-projected
                camera-optical points before gravity alignment.
      gravity   camera-frame down vector to use instead of GRAVITY_CAM.
      work_space  crop bbox dict {'x','y','z'} to use instead of WORK_SPACE."""
    R = R_G if gravity is None else gravity_rotation(np.asarray(gravity))
    ws = WORK_SPACE if work_space is None else work_space
    depth_m = depth_m[::stride, ::stride]
    v, u = np.indices(depth_m.shape, dtype=np.float32)
    valid = depth_m > 0
    z = depth_m[valid]
    x = (u[valid] * stride - K[0, 2]) * z / K[0, 0]
    y = (v[valid] * stride - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=-1)
    if T_delta is not None:
        T_delta = np.asarray(T_delta)
        pts_cam = pts_cam @ T_delta[:3, :3].T + T_delta[:3, 3]
    pc = pts_cam @ R.T

    m = ((pc[:, 0] > ws['x'][0]) & (pc[:, 0] < ws['x'][1]) &
         (pc[:, 1] > ws['y'][0]) & (pc[:, 1] < ws['y'][1]) &
         (pc[:, 2] > ws['z'][0]) & (pc[:, 2] < ws['z'][1]))
    pc = pc[m]
    if pc.shape[0] == 0:
        raise RuntimeError('empty point cloud after workspace crop')
    if pc.shape[0] > pre_sample:
        pc = pc[rng.choice(pc.shape[0], pre_sample, replace=False)]
    # FPS in float64: the training zarr was built this way and tie-breaking
    # in argmax depends on precision.
    sampled = farthest_point_sampling_torch(
        np.ascontiguousarray(pc, dtype=np.float64), num_points, device)
    return sampled.astype(np.float32)
