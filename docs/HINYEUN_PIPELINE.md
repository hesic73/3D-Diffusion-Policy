# Hinyeun glue-dispensing pipeline

How this fork is used with the `gluon` robot workstation repo
(`liang-group/dispensing` on the internal GitLab). Upstream 3D-Diffusion-Policy
docs cover the simulation benchmarks; everything Hinyeun-specific is here.

## End-to-end flow

```
gluon: collect demonstrations        (hand-guided or teleop, rosbags)
gluon: export LeRobot RGBD           (export_lerobot_rgbd, 30 Hz, 16-D state / 17-D action)
here:  convert_hinyeun_lerobot_to_zarr.py   LeRobot -> zarr (point clouds + agent_pos)
here:  train_policy.sh               with a hinyeun_*.yaml config
here:  export_deploy_checkpoint.py   EMA policy + hydra config only (small file)
here:  dp3_inference_server.py       TCP protocol v4, port 8890
gluon: scripts/rollout.sh            30 Hz control / 10 Hz action slots
```

Training configs (`3D-Diffusion-Policy/diffusion_policy_3d/config/`):

| Config | Action | Notes |
|---|---|---|
| `hinyeun_dp3.yaml` | 17-D bimanual absolute | |
| `hinyeun_delta_dp3.yaml` | 17-D bimanual chunk-delta | |
| `hinyeun_right_dp3.yaml` | 9-D right-only absolute | server slices state[8:16] |
| `hinyeun_right_delta_dp3.yaml` | 9-D right-only chunk-delta | |

Observation: 1024-point workspace cloud from the Orbbec depth (uint16 mm)
plus 16-D `agent_pos` = left arm 7 + left gripper width + right arm 7 +
right gripper width, exactly the order gluon's `policy_rollout` sends.

## Serving

```bash
conda activate dp3
python scripts/dp3_inference_server.py \
    --checkpoint <deploy>.ckpt \
    --camera-config migration/camera_delta_ep0_rgbd.json \
    --listen 0.0.0.0 --port 8890
```

`--camera-config` is mandatory: it supplies the live camera's
`gravity_cam_new` and `work_space_new` for point-cloud preprocessing.
`T_delta` in the same file is only for converting depths recorded before a
camera move (`estimate_camera_delta.py` produces it); it is never applied to
live observations. After a camera move, re-run the delta estimation and
restart the server with the new file.

Check the startup line before connecting a rollout: it must report the
expected `action_layout`/`action_dim`, and gluon's `policy_rollout` refuses
anything but `protocol_version 4`.

## Safety guards (do not weaken)

- **State distribution guard**: the server normalizes every incoming
  `agent_pos` with the checkpoint's own normalizer and rejects observations
  whose normalized value exceeds `--max-normalized-state` (default 1.25).
  Background: min-max normalization gives near-constant training dims (e.g.
  a left arm that never moved) scales in the thousands; a live pose 0.02 rad
  outside that range normalizes to +-80 and collapses the policy output to a
  fixed attractor pose regardless of the point cloud. This caused the
  2026-07-22 deployment failure (arms commanded 1.03 rad away from state).
  For future training runs prefer a larger normalizer `range_eps` (~0.01) or
  gaussian state normalization so the guard rarely triggers.
- **Do not re-apply the motion model at deployment.** gluon dataset actions
  already contain the right-arm LTI correction when recorded in dynamics
  replay mode.
- gluon-side guards (first-action distance, per-tick joint step clamp) live
  in `policy_rollout`; keep both layers.

## Diagnostics

- `python scripts/test_dp3_inference_server.py` — protocol unit tests, no GPU.
- `scripts/visualize_deploy_alignment.py` — offline train/deploy alignment
  and rollout safety report from a recorded rollout (depth + states) against
  the training zarr; renders normalized-state and cloud-chamfer charts.
  Use this before any first deployment of a new checkpoint.
- gluon's `ros2 run demonstration_manager policy_latency_probe` measures
  end-to-end and server-side inference latency against a live server.
