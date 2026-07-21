"""Estimate the Orbbec camera delta pose between the old dataset and the new
(post-bump) setup using RGB-D feature matching and PnP/RANSAC.

Run in the `dp3` conda env on the host (no ROS, no docker). Reads the original
LeRobot dataset READ-ONLY.

  python scripts/estimate_camera_delta.py \
      --root /path/to/hinyeun_glue_0714_lerobot_rgbd \
      --new-cloud new_cloud.npz --episode 0 --tick 0 \
      --out camera_delta.json --viz-out delta_viz.npz

Outputs camera_delta.json:
  T_delta      4x4 mapping OLD camera-optical points -> NEW camera-optical frame
  fitness, rmse_m, gravity_cam_new, work_space_new
"""
import argparse
import json
import os

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from hinyeun_preprocess import GRAVITY_CAM, WORK_SPACE, gravity_rotation

FPS_VIDEO = 30


def backproject(depth_m, K, color=None, stride=2, z_range=(0.15, 2.5)):
    depth_m = depth_m[::stride, ::stride]
    v, u = np.indices(depth_m.shape, dtype=np.float32)
    valid = (depth_m > z_range[0]) & (depth_m < z_range[1])
    z = depth_m[valid]
    x = (u[valid] * stride - K[0, 2]) * z / K[0, 0]
    y = (v[valid] * stride - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=-1).astype(np.float64)
    cols = None
    if color is not None:
        cols = (color[::stride, ::stride][valid].astype(np.float32) / 255.0)
    return pts, cols


def crop_to_workspace(pts, cols, margin):
    g = pts @ gravity_rotation(GRAVITY_CAM).T
    m = ((g[:, 0] > WORK_SPACE['x'][0] - margin) & (g[:, 0] < WORK_SPACE['x'][1] + margin) &
         (g[:, 1] > WORK_SPACE['y'][0] - margin) & (g[:, 1] < WORK_SPACE['y'][1] + margin) &
         (g[:, 2] > WORK_SPACE['z'][0] - margin) & (g[:, 2] < WORK_SPACE['z'][1] + margin))
    return pts[m], (cols[m] if cols is not None else None)


def remove_outliers(pts, cols, radius=0.02, min_neighbors=8, device="cuda"):
    """Radius outlier removal: drop points with too few neighbours (flying
    pixels / thin depth-edge streaks)."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    P = torch.tensor(pts, dtype=torch.float32, device=device)
    counts = torch.empty(len(P), device=device)
    for a in range(0, len(P), 2048):
        d = torch.cdist(P[a:a + 2048], P)
        counts[a:a + 2048] = (d < radius).sum(dim=1).float() - 1.0
    keep = (counts >= min_neighbors).cpu().numpy()
    return pts[keep], (cols[keep] if cols is not None else None)


def load_new_frame(npz_path):
    with np.load(npz_path) as data:
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        depth_m = data["depth_mm"].astype(np.float32) * 0.001
        color = data["color"].copy() if "color" in data else None
    return depth_m, K, color


def _decode_video_frame(root, info, row, video_key, want, dtype_scale=None):
    from_ts = float(row[f"videos/{video_key}/from_timestamp"])
    vpath = os.path.join(root, info["video_path"].format(
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"])))
    with av.open(vpath) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        container.seek(int(from_ts / tb), stream=stream, any_frame=False, backward=True)
        target = int(round(from_ts * FPS_VIDEO)) + want
        for frame in container.decode(stream):
            if int(round(float(frame.pts * tb) * FPS_VIDEO)) >= target:
                if dtype_scale is not None:
                    return frame.to_ndarray().astype(np.float32) * dtype_scale
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"could not decode {video_key} frame {want}")


def load_old_frame(root, episode, tick):
    with open(os.path.join(root, "meta/info.json")) as f:
        info = json.load(f)
    meta = info["features"]["observation.images.orbbec_depth"]["info"]
    m_per_code = float(meta["video.depth_max"]) / 4095.0
    ep_meta = pq.read_table(
        os.path.join(root, "meta/episodes/chunk-000/file-000.parquet")).to_pandas()
    row = ep_meta.iloc[episode]
    data_path = os.path.join(root, info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])))
    df = pq.read_table(data_path, columns=[
        "observation.camera.orbbec_intrinsics", "episode_index", "frame_index"]).to_pandas()
    df = df[df.episode_index == episode].sort_values("frame_index").reset_index(drop=True)
    K = np.asarray(df.iloc[tick]["observation.camera.orbbec_intrinsics"],
                   dtype=np.float64).reshape(3, 3)
    depth_m = _decode_video_frame(root, info, row, "observation.images.orbbec_depth",
                                  tick, dtype_scale=m_per_code)
    color = _decode_video_frame(root, info, row, "observation.images.orbbec", tick)
    return depth_m, K, color


def _camera_point(pixel, depth_m, K):
    u, v = np.rint(pixel).astype(int)
    z = float(depth_m[v, u])
    return [(u - K[0, 2]) * z / K[0, 0],
            (v - K[1, 2]) * z / K[1, 1], z]


def estimate_rgbd_pnp(old_depth, old_K, old_color,
                      new_depth, new_K, new_color,
                      ratio=0.7, reprojection_px=2.0):
    if old_color is None or new_color is None:
        raise RuntimeError("RGB-D PnP requires color in both the dataset and capture")
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01, edgeThreshold=15)
    old_keypoints, old_descriptors = sift.detectAndCompute(
        cv2.cvtColor(old_color, cv2.COLOR_RGB2GRAY), None)
    new_keypoints, new_descriptors = sift.detectAndCompute(
        cv2.cvtColor(new_color, cv2.COLOR_RGB2GRAY), None)
    if old_descriptors is None or new_descriptors is None:
        raise RuntimeError("SIFT found no RGB features")
    matches = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        old_descriptors, new_descriptors, k=2) if m.distance < ratio * n.distance]

    old_points = []
    new_pixels = []
    new_points = []
    old_h, old_w = old_depth.shape
    new_h, new_w = new_depth.shape
    for match in matches:
        old_pixel = old_keypoints[match.queryIdx].pt
        new_pixel = new_keypoints[match.trainIdx].pt
        old_u, old_v = np.rint(old_pixel).astype(int)
        new_u, new_v = np.rint(new_pixel).astype(int)
        if not (0 <= old_u < old_w and 0 <= old_v < old_h and
                0 <= new_u < new_w and 0 <= new_v < new_h):
            continue
        if not (0.2 < old_depth[old_v, old_u] < 2.0 and
                0.2 < new_depth[new_v, new_u] < 2.0):
            continue
        old_points.append(_camera_point(old_pixel, old_depth, old_K))
        new_pixels.append(new_pixel)
        new_points.append(_camera_point(new_pixel, new_depth, new_K))

    old_points = np.asarray(old_points, dtype=np.float64)
    new_pixels = np.asarray(new_pixels, dtype=np.float64)
    new_points = np.asarray(new_points, dtype=np.float64)
    if len(old_points) < 8:
        raise RuntimeError(f"Only {len(old_points)} valid RGB-D feature matches")
    ok, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        old_points, new_pixels, new_K, None,
        iterationsCount=10000, reprojectionError=reprojection_px,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 8:
        count = 0 if inliers is None else len(inliers)
        raise RuntimeError(f"RGB-D PnP found only {count} inliers")
    inliers = inliers[:, 0]
    cv2.solvePnP(old_points[inliers], new_pixels[inliers], new_K, None,
                 rotation_vector, translation, True, flags=cv2.SOLVEPNP_ITERATIVE)
    R = cv2.Rodrigues(rotation_vector)[0]
    t = translation[:, 0]
    projected = cv2.projectPoints(old_points[inliers], rotation_vector,
                                  translation, new_K, None)[0][:, 0]
    reprojection = np.linalg.norm(projected - new_pixels[inliers], axis=1)
    residual_3d = np.linalg.norm(
        old_points[inliers] @ R.T + t - new_points[inliers], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    metrics = {
        "feature_matches": len(old_points),
        "feature_inliers": len(inliers),
        "reprojection_rmse_px": float(np.sqrt(np.mean(reprojection ** 2))),
        "rgbd_rmse_m": float(np.sqrt(np.mean(residual_3d ** 2))),
        "rgbd_median_m": float(np.median(residual_3d)),
    }
    return T, metrics


def workspace_new(T_delta):
    R_d, t_d = T_delta[:3, :3], T_delta[:3, 3]
    R_g_old = gravity_rotation(GRAVITY_CAM)
    g_new = R_d @ GRAVITY_CAM
    R_g_new = gravity_rotation(g_new)
    A, b = R_g_new @ R_d @ R_g_old.T, R_g_new @ t_d
    corners = np.array([[x, y, z] for x in WORK_SPACE['x'] for y in WORK_SPACE['y']
                        for z in WORK_SPACE['z']])
    moved = corners @ A.T + b
    return g_new, {ax: [float(moved[:, i].min()), float(moved[:, i].max())]
                   for i, ax in enumerate("xyz")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--new-cloud", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--tick", type=int, default=0)
    ap.add_argument("--out", default="camera_delta.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-ratio", type=float, default=0.7)
    ap.add_argument("--reprojection-px", type=float, default=2.0)
    ap.add_argument("--crop-margin", type=float, default=0.15)
    ap.add_argument("--no-clean", action="store_true", help="skip outlier removal")
    ap.add_argument("--viz-out", default=None)
    args = ap.parse_args()

    old_depth, old_K, old_c_image = load_old_frame(args.root, args.episode, args.tick)
    new_depth, new_K, new_c_image = load_new_frame(args.new_cloud)
    T, metrics = estimate_rgbd_pnp(
        old_depth, old_K, old_c_image, new_depth, new_K, new_c_image,
        ratio=args.feature_ratio, reprojection_px=args.reprojection_px)
    fitness = metrics["feature_inliers"] / metrics["feature_matches"]
    rmse = metrics["rgbd_rmse_m"]
    print(f"RGB-D PnP: {metrics['feature_inliers']}/"
          f"{metrics['feature_matches']} inliers, reprojection "
          f"RMSE={metrics['reprojection_rmse_px']:.2f}px, 3D median="
          f"{metrics['rgbd_median_m'] * 1000:.2f}mm")
    g_new, ws_new = workspace_new(T)
    angle = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    print(f"fitness={fitness:.3f} rmse={rmse * 1000:.2f}mm "
          f"angle={angle:.1f}deg trans={np.linalg.norm(T[:3, 3]) * 100:.1f}cm")
    print("T_delta:\n", np.array2string(T, precision=5))

    out = {"T_delta": T.tolist(), "method": "rgbd-pnp",
           "fitness": fitness, "rmse_m": rmse, **metrics,
           "gravity_cam_new": g_new.tolist(), "work_space_new": ws_new,
           "source": {"root": args.root, "episode": args.episode, "tick": args.tick,
                      "new_cloud": args.new_cloud}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    if args.viz_out:
        old_p, old_c = backproject(old_depth, old_K, old_c_image)
        new_p, new_c = backproject(new_depth, new_K, new_c_image)
        old_p, old_c = crop_to_workspace(old_p, old_c, args.crop_margin)
        new_p, new_c = crop_to_workspace(new_p, new_c, args.crop_margin)
        if not args.no_clean:
            old_p, old_c = remove_outliers(old_p, old_c, device=args.device)
            new_p, new_c = remove_outliers(new_p, new_c, device=args.device)
        rng = np.random.default_rng(1)

        def sub(p, c, n=8000):
            i = rng.choice(len(p), min(n, len(p)), replace=False)
            return p[i].astype(np.float32), (c[i].astype(np.float32)
                                             if c is not None else np.zeros((len(i), 3), np.float32))

        op, oc = sub(old_p, old_c)
        npp, nc = sub(new_p, new_c)
        np.savez(args.viz_out, old=op, old_rgb=oc, new=npp, new_rgb=nc, T_delta=T)
        print(f"wrote {args.viz_out}")


if __name__ == "__main__":
    main()
