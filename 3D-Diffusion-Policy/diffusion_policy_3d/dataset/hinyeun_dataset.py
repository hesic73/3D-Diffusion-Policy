from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class HinyeunDataset(BaseDataset):
    SCHEMAS = {
        'bimanual': {
            'state_indices': np.arange(16),
            'action_indices': np.arange(17),
            'joint_slices': (
                (slice(0, 7), slice(0, 7)),
                (slice(8, 15), slice(8, 15)),
            ),
        },
        'right_only': {
            'state_indices': np.arange(8, 16),
            'action_indices': np.arange(8, 17),
            'joint_slices': ((slice(0, 7), slice(0, 7)),),
        },
    }

    def __init__(self,
            zarr_path,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            action_representation='absolute',
            action_schema='bimanual',
            ):
        super().__init__()
        if action_representation not in ('absolute', 'chunk_delta_qpos'):
            raise ValueError(
                f'Unsupported action representation: {action_representation}')
        if not 1 <= n_obs_steps <= horizon:
            raise ValueError(
                f'n_obs_steps must be in [1, horizon], got {n_obs_steps}')
        if action_schema not in self.SCHEMAS:
            raise ValueError(f'Unsupported action schema: {action_schema}')
        self.task_name = task_name
        self.action_representation = action_representation
        self.action_schema = action_schema
        schema = self.SCHEMAS[action_schema]
        self.state_indices = schema['state_indices']
        self.action_indices = schema['action_indices']
        self.joint_slices = schema['joint_slices']
        self.n_obs_steps = n_obs_steps
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        action_data = self.replay_buffer['action'][:, self.action_indices]
        if self.action_representation == 'chunk_delta_qpos':
            # The delta target depends on each sequence's current observation,
            # so fit its statistics over training windows rather than raw rows.
            action_sampler = SequenceSampler(
                replay_buffer=self.replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                keys=['state', 'action'],
                episode_mask=self.train_mask)
            action_data = np.empty(
                (len(action_sampler), self.horizon, len(self.action_indices)),
                dtype=np.float32)
            for idx in range(len(action_sampler)):
                sample = action_sampler.sample_sequence(idx)
                action_data[idx] = self._encode_action(
                    sample['action'], sample['state'])

        data = {
            'action': action_data,
            'agent_pos': self.replay_buffer['state'][:, self.state_indices],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _encode_action(self, action, agent_pos):
        action = action[:, self.action_indices].astype(np.float32, copy=True)
        if self.action_representation == 'absolute':
            return action

        state = agent_pos[:, self.state_indices]
        q_ref = state[self.n_obs_steps - 1]
        for action_slice, state_slice in self.joint_slices:
            action[:, action_slice] -= q_ref[state_slice]
        return action

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, self.state_indices].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 3
                'agent_pos': agent_pos,
            },
            'action': self._encode_action(
                sample['action'], sample['state'])
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
