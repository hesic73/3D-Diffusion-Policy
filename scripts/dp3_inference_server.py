"""TCP inference server for deploying the hinyeun glue DP3 policy.

Runs on the host in the `dp3` conda env (GPU). The `policy_rollout` ROS node
inside the robot container connects over TCP and streams observations; this
server applies the exact training-time preprocessing (gravity alignment,
workspace crop, FPS to 1024 points) and returns DP3 action chunks.

Framing (both directions): 4-byte big-endian header length, JSON header,
then `payload_bytes` of raw payload as declared in the header.

Client -> server:
  {"type": "reset"}                                  # begin an episode
  {"type": "info"}                                   # checkpoint + chunk shape
  {"type": "infer", "n_observations": N,
   "states": [[16 floats] * N], "intrinsics": [[9 floats] * N],
   "height": H, "width": W, "payload_bytes": N*H*W*2}
                                                       # + N uint16 depth frames
Server -> client:
  {"type": "ok"} | {"type": "info", "protocol_version": 4,
   "action_layout": "bimanual" | "right_only", "action_dim": 17 | 9, ...} |
  {"type": "action", "actions": [[action_dim floats] * n_action_steps]} |
  {"type": "error", "message": "..."}

Usage:
  python scripts/dp3_inference_server.py \
      --checkpoint .../checkpoints/latest.ckpt \
      --camera-config migration/camera_delta_ep0_rgbd.json \
      --listen 0.0.0.0 --port 8890
"""
import argparse
import json
import socket
import socketserver
import struct
import sys
import time

import numpy as np
import torch

from hinyeun_preprocess import depth_m_to_workspace_cloud


def load_camera_config(path):
    with open(path) as stream:
        value = json.load(stream)
    gravity = np.asarray(value['gravity_cam_new'], dtype=np.float64)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError('gravity_cam_new must contain three finite values')
    norm = float(np.linalg.norm(gravity))
    if not 0.9 <= norm <= 1.1:
        raise ValueError(
            f'gravity_cam_new must be a unit vector, got norm {norm:.6f}')
    workspace = {}
    for axis in ('x', 'y', 'z'):
        bounds = np.asarray(value['work_space_new'][axis], dtype=np.float64)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1]):
            raise ValueError(
                f'work_space_new.{axis} must be two increasing finite values')
        workspace[axis] = (float(bounds[0]), float(bounds[1]))
    return gravity, workspace


def agent_pos_normalizer(policy):
    params = policy.normalizer['agent_pos'].params_dict
    scale = params['scale'].detach().cpu().numpy().astype(np.float32)
    offset = params['offset'].detach().cpu().numpy().astype(np.float32)
    if scale.ndim != 1 or offset.shape != scale.shape:
        raise ValueError('checkpoint agent_pos normalizer has invalid shape')
    if not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise ValueError('checkpoint agent_pos normalizer is not finite')
    if np.any(scale == 0):
        raise ValueError('checkpoint agent_pos normalizer has zero scale')
    return scale, offset


def load_policy(checkpoint_path, device):
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent / '3D-Diffusion-Policy'
    sys.path.insert(0, str(repo))
    import dill
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)

    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill,
                         map_location='cpu', weights_only=False)
    cfg = payload['cfg']
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload['state_dicts']['ema_model'])
    policy.eval().to(device)
    return policy, cfg


class PolicySession:
    def __init__(self, policy, cfg, device, gravity, workspace,
                 max_normalized_state=1.25):
        self.policy = policy
        self.device = device
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.action_layout = policy.action_schema
        self.action_dim = int(policy.action_dim)
        expected_dimensions = {
            'bimanual': (16, 17),
            'right_only': (8, 9),
        }
        if self.action_layout not in expected_dimensions:
            raise ValueError(f'unsupported action layout: {self.action_layout}')
        self.state_dim, expected_action_dim = expected_dimensions[
            self.action_layout]
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f'{self.action_layout} policy must use {expected_action_dim}-D '
                f'actions, got {self.action_dim}')
        self.rng = np.random.default_rng()
        self.gravity = gravity
        self.workspace = workspace
        self.max_normalized_state = float(max_normalized_state)
        if np.isfinite(self.max_normalized_state):
            self.state_scale, self.state_offset = agent_pos_normalizer(policy)
            if self.state_scale.shape != (self.state_dim,):
                raise ValueError(
                    'checkpoint agent_pos normalizer dimension does not match '
                    f'{self.action_layout}: {self.state_scale.shape}')
        else:
            self.state_scale = self.state_offset = None

    def validate_states(self, states):
        if self.state_scale is None:
            return
        normalized = states * self.state_scale + self.state_offset
        bad = np.argwhere(np.abs(normalized) > self.max_normalized_state)
        if bad.size == 0:
            return
        row, index = (int(v) for v in bad[0])
        scale = float(self.state_scale[index])
        offset = float(self.state_offset[index])
        low = min((-1.0 - offset) / scale, (1.0 - offset) / scale)
        high = max((-1.0 - offset) / scale, (1.0 - offset) / scale)
        raise ValueError(
            'agent_pos is outside the checkpoint training distribution: '
            f'observation {row}, state[{index}]={states[row, index]:.6f}, '
            f'training range=[{low:.6f}, {high:.6f}], '
            f'normalized={normalized[row, index]:.2f}, '
            f'limit={self.max_normalized_state:.2f}')

    def infer(self, states, depths_mm, intrinsics):
        t0 = time.monotonic()
        states = np.asarray(states, dtype=np.float32)
        depths_mm = np.asarray(depths_mm, dtype=np.uint16)
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        count = states.shape[0]
        if states.shape != (count, 16):
            raise ValueError(f'wire states must be (N,16), got {states.shape}')
        if depths_mm.ndim != 3 or depths_mm.shape[0] != count:
            raise ValueError(f'depths must be (N,H,W), got {depths_mm.shape}')
        if intrinsics.shape != (count, 3, 3):
            raise ValueError(f'intrinsics must be (N,3,3), got {intrinsics.shape}')
        if not 1 <= count <= self.n_obs_steps:
            raise ValueError(
                f'expected 1..{self.n_obs_steps} observations, got {count}')
        if self.action_layout == 'right_only':
            states = states[:, 8:16]
        if not np.isfinite(states).all():
            raise ValueError('agent_pos contains non-finite values')
        self.validate_states(states)

        history = []
        for state, depth_mm, K in zip(states, depths_mm, intrinsics):
            depth_m = depth_mm.astype(np.float32) * 0.001
            pcd = depth_m_to_workspace_cloud(
                depth_m, K, self.rng, device=self.device,
                gravity=self.gravity, work_space=self.workspace)
            history.append((pcd, state))
        history = [history[0]] * (self.n_obs_steps - count) + history
        obs = {
            'point_cloud': torch.from_numpy(
                np.stack([h[0] for h in history])[None]).to(self.device),
            'agent_pos': torch.from_numpy(
                np.stack([h[1] for h in history])[None]).to(self.device),
        }
        with torch.no_grad():
            result = self.policy.predict_action(obs)
        actions = result['action'][0].cpu().numpy()
        assert actions.shape == (self.n_action_steps, self.action_dim), actions.shape
        return actions, time.monotonic() - t0


def read_frame(rfile):
    header_len_bytes = rfile.read(4)
    if len(header_len_bytes) < 4:
        return None, None
    (header_len,) = struct.unpack('>I', header_len_bytes)
    header = json.loads(rfile.read(header_len))
    payload = b''
    n = int(header.get('payload_bytes', 0))
    if n:
        payload = rfile.read(n)
        if len(payload) < n:
            return None, None
    return header, payload


def write_frame(wfile, header, payload=b''):
    header = dict(header, payload_bytes=len(payload))
    raw = json.dumps(header).encode()
    wfile.write(struct.pack('>I', len(raw)) + raw + payload)
    wfile.flush()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        session = PolicySession(
            server.policy, server.cfg, server.device,
            server.gravity, server.workspace, server.max_normalized_state)
        server.log(f'client connected: {self.client_address}')
        while True:
            try:
                header, payload = read_frame(self.rfile)
            except (ConnectionError, json.JSONDecodeError) as exc:
                server.log(f'client dropped: {exc}')
                return
            if header is None:
                server.log('client disconnected')
                return
            try:
                kind = header['type']
                if kind == 'reset':
                    write_frame(self.wfile, {'type': 'ok'})
                elif kind == 'info':
                    write_frame(self.wfile, {
                        'type': 'info',
                        'protocol_version': 4,
                        'checkpoint': server.checkpoint,
                        'horizon': int(server.cfg.horizon),
                        'n_obs_steps': session.n_obs_steps,
                        'n_action_steps': session.n_action_steps,
                        'action_layout': session.action_layout,
                        'action_dim': session.action_dim,
                        'camera_config': server.camera_config,
                        'max_normalized_state': server.max_normalized_state,
                    })
                elif kind == 'infer':
                    count = int(header['n_observations'])
                    states = np.asarray(header['states'], dtype=np.float32)
                    intrinsics = np.asarray(
                        header['intrinsics'], dtype=np.float64).reshape(
                            count, 3, 3)
                    h, w = int(header['height']), int(header['width'])
                    depths = np.frombuffer(payload, dtype='<u2').reshape(
                        count, h, w)
                    actions, dt = session.infer(states, depths, intrinsics)
                    write_frame(self.wfile, {
                        'type': 'action',
                        'actions': [[float(v) for v in row] for row in actions],
                        'inference_ms': round(dt * 1000, 1),
                    })
                    server.log(f'infer {dt*1000:.0f} ms')
                else:
                    raise ValueError(f'unknown message type: {kind}')
            except Exception as exc:
                write_frame(self.wfile, {'type': 'error', 'message': str(exc)})
                server.log(f'error: {exc}')


class InferenceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, policy, cfg, device, verbose, checkpoint,
                 camera_config, gravity, workspace, max_normalized_state):
        super().__init__(address, Handler)
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.camera_config = camera_config
        self.gravity = gravity
        self.workspace = workspace
        self.max_normalized_state = max_normalized_state

    def log(self, message):
        if self.verbose:
            print(f'[dp3-server] {message}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument(
        '--camera-config', required=True,
        help='Accepted new-camera JSON containing gravity_cam_new and '
             'work_space_new. T_delta is intentionally not applied live.')
    ap.add_argument(
        '--max-normalized-state', type=float, default=1.25,
        help='Reject agent_pos outside this absolute normalized value.')
    ap.add_argument('--listen', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8890)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    if args.max_normalized_state < 1.0:
        ap.error('--max-normalized-state must be at least 1.0')

    gravity, workspace = load_camera_config(args.camera_config)

    print(f'loading checkpoint {args.checkpoint} ...', flush=True)
    policy, cfg = load_policy(args.checkpoint, args.device)
    print(f'policy ready: n_obs_steps={cfg.n_obs_steps} '
          f'n_action_steps={cfg.n_action_steps} horizon={cfg.horizon} '
          f'action_layout={policy.action_schema} action_dim={policy.action_dim}',
          flush=True)

    # warm up cuda kernels so the first real request is not slow
    warm = PolicySession(
        policy, cfg, args.device, gravity, workspace,
        args.max_normalized_state)
    fake_depth = np.full((800, 1280), 900, dtype=np.uint16)
    fake_k = np.array([[611.0, 0, 640.9], [0, 610.9, 407.4], [0, 0, 1.0]])
    scale, offset = agent_pos_normalizer(policy)
    mean_state = -offset / scale
    if policy.action_schema == 'right_only':
        wire_state = np.zeros(16, dtype=np.float32)
        wire_state[8:16] = mean_state
    else:
        wire_state = mean_state
    _, dt = warm.infer(
        wire_state[None],
        fake_depth[None],
        fake_k[None],
    )
    print(f'warmup inference: {dt*1000:.0f} ms', flush=True)

    server = InferenceServer((args.listen, args.port), policy, cfg,
                             args.device, not args.quiet, str(args.checkpoint),
                             str(args.camera_config), gravity, workspace,
                             args.max_normalized_state)
    print(f'listening on {args.listen}:{args.port}', flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
