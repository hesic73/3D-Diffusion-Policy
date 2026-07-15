"""Convert the hinyeun glue LeRobot v3 RGBD dataset to a DP3 ReplayBuffer zarr.

Pipeline per policy tick (subsampled from 30 Hz to 10 Hz, matching the Orbbec
depth rate): decode the registered Orbbec depth frame, back-project with the
per-frame intrinsics, rotate into a gravity-aligned frame (z up, from
meta/orbbec_gravity.yaml), crop to the workspace bbox, and farthest-point
sample to a fixed number of points.

Output zarr layout (DP3 ReplayBuffer):
  data/state       (N, 16) float32
  data/action      (N, 17) float32
  data/point_cloud (N, num_points, 3) float32   # gravity-aligned xyz, meters
  meta/episode_ends (E,) int64

Usage:
  python scripts/convert_hinyeun_lerobot_to_zarr.py \
      --root /home/hsc/Downloads/hinyeun_glue_0714_lerobot_rgbd \
      --out 3D-Diffusion-Policy/data/hinyeun_glue.zarr
"""
import argparse
import json
import os

import av
import numpy as np
import pyarrow.parquet as pq
import zarr
from termcolor import cprint

from hinyeun_preprocess import depth_m_to_workspace_cloud

FPS_VIDEO = 30


def depth_m_per_code(info):
    """12-bit codes span [0, video.depth_max] metres; the 0714 dataset used
    depth_max=5.0 (verified against lerobot 0.6.0 decoding), newer exporter
    versions use 4.095 (code = mm)."""
    meta = info['features']['observation.images.orbbec_depth']['info']
    if meta.get('video.use_log') or meta.get('video.shift'):
        raise ValueError('log/shift depth encodings are not supported')
    return float(meta['video.depth_max']) / 4095.0


def depth_to_workspace_cloud(depth_codes, K, stride, rng, m_per_code,
                             num_points=1024, device='cuda'):
    depth_m = depth_codes.astype(np.float32) * m_per_code
    return depth_m_to_workspace_cloud(
        depth_m, K, rng, stride=stride, num_points=num_points, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--subsample', type=int, default=3, help='take every k-th 30Hz tick')
    ap.add_argument('--num-points', type=int, default=1024)
    ap.add_argument('--stride', type=int, default=2, help='depth image pixel stride')
    ap.add_argument('--episodes', type=int, default=None, help='limit number of episodes (debug)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    root = args.root
    rng = np.random.default_rng(42)

    with open(os.path.join(root, 'meta/info.json')) as f:
        info = json.load(f)
    assert info['codebase_version'] == 'v3.0'

    m_per_code = depth_m_per_code(info)
    cprint(f'depth scale: {m_per_code * 1000:.6f} mm per 12-bit code', 'cyan')

    ep_meta = pq.read_table(
        os.path.join(root, 'meta/episodes/chunk-000/file-000.parquet')).to_pandas()
    n_episodes = len(ep_meta) if args.episodes is None else min(args.episodes, len(ep_meta))

    state_out, action_out, pcd_out, episode_ends = [], [], [], []
    total = 0

    for ep_i in range(n_episodes):
        row = ep_meta.iloc[ep_i]
        assert int(row['episode_index']) == ep_i
        dchunk, dfile = int(row['data/chunk_index']), int(row['data/file_index'])
        data_path = os.path.join(root, info['data_path'].format(chunk_index=dchunk, file_index=dfile))
        df = pq.read_table(data_path, columns=[
            'observation.state', 'action', 'observation.camera.orbbec_intrinsics',
            'episode_index', 'frame_index']).to_pandas()
        df = df[df.episode_index == ep_i].sort_values('frame_index').reset_index(drop=True)
        assert len(df) == int(row['length'])

        vchunk = int(row['videos/observation.images.orbbec_depth/chunk_index'])
        vfile = int(row['videos/observation.images.orbbec_depth/file_index'])
        from_ts = float(row['videos/observation.images.orbbec_depth/from_timestamp'])
        vpath = os.path.join(root, info['video_path'].format(
            video_key='observation.images.orbbec_depth', chunk_index=vchunk, file_index=vfile))

        picked = list(range(0, len(df), args.subsample))
        first_vframe = int(round(from_ts * FPS_VIDEO))
        wanted = {first_vframe + t: t for t in picked}

        K = np.array(df.iloc[0]['observation.camera.orbbec_intrinsics'],
                     dtype=np.float64).reshape(3, 3)

        got = 0
        with av.open(vpath) as container:
            stream = container.streams.video[0]
            tb = stream.time_base
            container.seek(int(from_ts / tb), stream=stream, any_frame=False, backward=True)
            for frame in container.decode(stream):
                vidx = int(round(float(frame.pts * tb) * FPS_VIDEO))
                if vidx in wanted:
                    t = wanted[vidx]
                    pcd = depth_to_workspace_cloud(
                        frame.to_ndarray(), K, args.stride, rng, m_per_code,
                        num_points=args.num_points, device=args.device)
                    pcd_out.append(pcd.astype(np.float32))
                    state_out.append(np.asarray(df.iloc[t]['observation.state'], dtype=np.float32))
                    action_out.append(np.asarray(df.iloc[t]['action'], dtype=np.float32))
                    got += 1
                if vidx >= first_vframe + picked[-1]:
                    break
        if got != len(picked):
            raise RuntimeError(f'episode {ep_i}: decoded {got}/{len(picked)} frames')
        total += got
        episode_ends.append(total)
        cprint(f'episode {ep_i}: {got} ticks (total {total})', 'green')

    state_arr = np.stack(state_out)
    action_arr = np.stack(action_out)
    pcd_arr = np.stack(pcd_out)
    ee_arr = np.array(episode_ends, dtype=np.int64)

    if os.path.exists(args.out):
        cprint(f'Overwriting {args.out}', 'red')
        os.system(f'rm -rf {args.out}')
    os.makedirs(args.out, exist_ok=True)

    zroot = zarr.group(args.out)
    zdata = zroot.create_group('data')
    zmeta = zroot.create_group('meta')
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    zdata.create_dataset('state', data=state_arr, chunks=(1000, state_arr.shape[1]),
                         dtype='float32', compressor=compressor)
    zdata.create_dataset('action', data=action_arr, chunks=(1000, action_arr.shape[1]),
                         dtype='float32', compressor=compressor)
    zdata.create_dataset('point_cloud', data=pcd_arr,
                         chunks=(100, pcd_arr.shape[1], pcd_arr.shape[2]),
                         dtype='float32', compressor=compressor)
    zmeta.create_dataset('episode_ends', data=ee_arr, chunks=(1000,),
                         dtype='int64', compressor=compressor)

    cprint(f'state {state_arr.shape}  action {action_arr.shape}  '
           f'point_cloud {pcd_arr.shape}  episodes {len(ee_arr)}', 'cyan')
    cprint(f'saved to {args.out}', 'cyan')


if __name__ == '__main__':
    main()
