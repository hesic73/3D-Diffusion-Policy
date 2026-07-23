"""Create an offline train/deploy alignment and rollout safety report."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from dp3_inference_server import agent_pos_normalizer, load_policy


SURFACE = '#fcfcfb'
INK = '#171717'
GRID = '#dfded9'
TRAIN = '#5966b2'
DEPLOY = '#d44735'
FIXED = '#288f58'


def symmetric_nn_mean(left, right):
    def nearest_mean(source, target):
        values = []
        for start in range(0, len(source), 256):
            delta = source[start:start + 256, None] - target[None]
            values.append(np.sqrt(np.square(delta).sum(axis=2)).min(axis=1))
        return np.concatenate(values).mean()

    left_to_right = nearest_mean(left, right)
    right_to_left = nearest_mean(right, left)
    return float((left_to_right + right_to_left) / 2.0)


def infer(policy, n_obs_steps, cloud, state, device):
    observation = {
        'point_cloud': torch.from_numpy(
            np.repeat(cloud[None, None], n_obs_steps, axis=1)
        ).to(device=device, dtype=torch.float32),
        'agent_pos': torch.from_numpy(
            np.repeat(state[None, None], n_obs_steps, axis=1)
        ).to(device=device, dtype=torch.float32),
    }
    with torch.no_grad():
        return policy.predict_action(observation)['action'][0].cpu().numpy()


def style_axis(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.7)
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color(GRID)


def plot_cloud_report(data, train_cloud, output):
    old_distance = symmetric_nn_mean(data['cloud_old'], train_cloud)
    new_distance = symmetric_nn_mean(data['cloud_new'], train_cloud)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), facecolor=SURFACE)
    for column, (name, cloud, color, distance) in enumerate((
        ('Deployed preprocessing', data['cloud_old'], DEPLOY, old_distance),
        ('Corrected preprocessing', data['cloud_new'], FIXED, new_distance),
    )):
        for row, (horizontal, vertical, labels) in enumerate((
            (0, 1, ('x', 'y')),
            (0, 2, ('x', 'z')),
        )):
            axis = axes[row, column]
            axis.scatter(
                train_cloud[:, horizontal], train_cloud[:, vertical],
                s=4, c=TRAIN, alpha=0.38, linewidths=0,
                label='nearest training cloud',
            )
            axis.scatter(
                cloud[:, horizontal], cloud[:, vertical],
                s=4, c=color, alpha=0.48, linewidths=0,
                label='live observation',
            )
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlabel(f'{labels[0]} (m)')
            axis.set_ylabel(f'{labels[1]} (m)')
            axis.set_title(
                f'{name}: {labels[0].upper()}{labels[1].upper()}\n'
                f'symmetric NN mean = {distance * 1000:.1f} mm',
                color=INK,
            )
            style_axis(axis)
            if row == 0:
                axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        'Live point cloud versus nearest retargeted training observation',
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return old_distance, new_distance


def plot_safety_report(data, normalized, diagnostic, output, limit):
    labels = [f'L{i}' for i in range(1, 8)] + ['LG'] + [
        f'R{i}' for i in range(1, 8)
    ] + ['RG']
    recorded = data['recorded']
    right_now = data['right_now']
    times = np.arange(recorded.shape[0]) / 10.0

    fig = plt.figure(figsize=(11, 12), facecolor=SURFACE)
    grid = fig.add_gridspec(8, 1, height_ratios=[1.7] + [1] * 7)
    axis = fig.add_subplot(grid[0])
    magnitudes = np.maximum(np.abs(normalized), 1e-3)
    colors = [DEPLOY if value > limit else TRAIN for value in magnitudes]
    axis.bar(labels, magnitudes, color=colors)
    axis.axhline(limit, color=INK, linestyle='--', linewidth=1.2,
                 label=f'server rejection limit ({limit:g})')
    axis.set_yscale('log')
    axis.set_ylabel('|normalized state| (log scale)')
    axis.set_title(
        'Failed live state against checkpoint normalization '
        '(red dimensions are OOD)', color=INK,
    )
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    for joint in range(7):
        axis = fig.add_subplot(grid[joint + 1])
        axis.plot(
            times, recorded[:, 8 + joint], color=DEPLOY, linewidth=2,
            label='recorded deployed chunk' if joint == 0 else None,
        )
        axis.plot(
            times, diagnostic[:, 8 + joint], color=FIXED, linewidth=2,
            label='offline clamped-state diagnostic' if joint == 0 else None,
        )
        axis.axhline(
            right_now[joint], color=INK, linestyle='--', linewidth=1,
            label='observed robot position' if joint == 0 else None,
        )
        axis.set_ylabel(f'R{joint + 1}\nrad', rotation=0, labelpad=24)
        style_axis(axis)
        if joint == 0:
            axis.legend(frameon=False, fontsize=8, ncol=3)
        if joint != 6:
            axis.tick_params(labelbottom=False)
    axis.set_xlabel('chunk time (s; policy action slots at 10 Hz)')
    fig.suptitle(
        'Why the failed rollout moved dangerously\n'
        'The clamped curve is diagnostic only; deployment rejects OOD states',
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repro', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--zarr', type=Path, required=True)
    parser.add_argument('--train-index', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--normalized-limit', type=float, default=1.25)
    args = parser.parse_args()

    data = np.load(args.repro)
    dataset = zarr.open(str(args.zarr), mode='r')
    train_cloud = np.asarray(dataset['data/point_cloud'][args.train_index])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy, cfg = load_policy(args.checkpoint, args.device)
    scale, offset = agent_pos_normalizer(policy)
    state = np.asarray(data['state'], dtype=np.float32)
    normalized = state * scale + offset
    low = np.minimum((-1.0 - offset) / scale, (1.0 - offset) / scale)
    high = np.maximum((-1.0 - offset) / scale, (1.0 - offset) / scale)
    clamped_state = np.clip(state, low, high).astype(np.float32)
    diagnostic = infer(
        policy, int(cfg.n_obs_steps), data['cloud_new'], clamped_state,
        args.device,
    )

    cloud_path = args.output_dir / 'pointcloud_alignment.png'
    safety_path = args.output_dir / 'rollout_safety.png'
    old_distance, new_distance = plot_cloud_report(
        data, train_cloud, cloud_path
    )
    plot_safety_report(
        data, normalized, diagnostic, safety_path, args.normalized_limit
    )
    print(f'wrote {cloud_path}')
    print(f'wrote {safety_path}')
    print(f'deployed cloud symmetric NN mean: {old_distance * 1000:.1f} mm')
    print(f'corrected cloud symmetric NN mean: {new_distance * 1000:.1f} mm')
    print(f'max |normalized live state|: {np.abs(normalized).max():.1f}')


if __name__ == '__main__':
    main()
