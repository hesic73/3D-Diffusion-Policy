"""TCP inference server for deploying the hinyeun glue DP3 policy.

Runs on the host in the `dp3` conda env (GPU). The `policy_rollout` ROS node
inside the robot container connects over TCP and streams observations; this
server applies the exact training-time preprocessing (gravity alignment,
workspace crop, FPS to 1024 points) and returns DP3 action chunks.

Framing (both directions): 4-byte big-endian header length, JSON header,
then `payload_bytes` of raw payload as declared in the header.

Client -> server:
  {"type": "reset"}                                  # clear obs history
  {"type": "infer", "state": [16 floats],
   "k": [9 floats, row-major 3x3 intrinsics],
   "height": H, "width": W, "payload_bytes": H*W*2}  # + uint16 depth in mm
Server -> client:
  {"type": "ok"}                                     # reset ack
  {"type": "action", "actions": [[17 floats] * n_action_steps]}
  {"type": "error", "message": "..."}

Usage:
  python scripts/dp3_inference_server.py \
      --checkpoint .../checkpoints/latest.ckpt --listen 0.0.0.0 --port 8890
"""
import argparse
import json
import socket
import socketserver
import struct
import sys
import time
from collections import deque

import numpy as np
import torch

from hinyeun_preprocess import depth_m_to_workspace_cloud


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
    def __init__(self, policy, cfg, device):
        self.policy = policy
        self.device = device
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.rng = np.random.default_rng()
        self.history = deque(maxlen=self.n_obs_steps)

    def reset(self):
        self.history.clear()

    def infer(self, state, depth_mm, K):
        t0 = time.monotonic()
        depth_m = depth_mm.astype(np.float32) * 0.001
        pcd = depth_m_to_workspace_cloud(depth_m, K, self.rng, device=self.device)
        self.history.append((pcd, state))
        while len(self.history) < self.n_obs_steps:
            self.history.appendleft(self.history[0])
        obs = {
            'point_cloud': torch.from_numpy(
                np.stack([h[0] for h in self.history])[None]).to(self.device),
            'agent_pos': torch.from_numpy(
                np.stack([h[1] for h in self.history])[None]).to(self.device),
        }
        with torch.no_grad():
            result = self.policy.predict_action(obs)
        actions = result['action'][0].cpu().numpy()
        assert actions.shape == (self.n_action_steps, 17), actions.shape
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
        session = PolicySession(server.policy, server.cfg, server.device)
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
                    session.reset()
                    write_frame(self.wfile, {'type': 'ok'})
                elif kind == 'info':
                    write_frame(self.wfile, {
                        'type': 'info',
                        'checkpoint': server.checkpoint,
                        'n_obs_steps': session.n_obs_steps,
                        'n_action_steps': session.n_action_steps,
                    })
                elif kind == 'infer':
                    state = np.asarray(header['state'], dtype=np.float32)
                    if state.shape != (16,):
                        raise ValueError(f'state must be 16-D, got {state.shape}')
                    K = np.asarray(header['k'], dtype=np.float64).reshape(3, 3)
                    h, w = int(header['height']), int(header['width'])
                    depth = np.frombuffer(payload, dtype='<u2').reshape(h, w)
                    actions, dt = session.infer(state, depth, K)
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

    def __init__(self, address, policy, cfg, device, verbose, checkpoint):
        super().__init__(address, Handler)
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.verbose = verbose
        self.checkpoint = checkpoint

    def log(self, message):
        if self.verbose:
            print(f'[dp3-server] {message}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--listen', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8890)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    print(f'loading checkpoint {args.checkpoint} ...', flush=True)
    policy, cfg = load_policy(args.checkpoint, args.device)
    print(f'policy ready: n_obs_steps={cfg.n_obs_steps} '
          f'n_action_steps={cfg.n_action_steps} horizon={cfg.horizon}', flush=True)

    # warm up cuda kernels so the first real request is not slow
    warm = PolicySession(policy, cfg, args.device)
    fake_depth = np.full((800, 1280), 900, dtype=np.uint16)
    fake_k = np.array([[611.0, 0, 640.9], [0, 610.9, 407.4], [0, 0, 1.0]])
    _, dt = warm.infer(np.zeros(16, dtype=np.float32), fake_depth, fake_k)
    print(f'warmup inference: {dt*1000:.0f} ms', flush=True)

    server = InferenceServer((args.listen, args.port), policy, cfg,
                             args.device, not args.quiet, str(args.checkpoint))
    print(f'listening on {args.listen}:{args.port}', flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
