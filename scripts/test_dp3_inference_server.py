from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp3_inference_server as inference_server
from diffusion_policy_3d.policy.dp3 import DP3


class FakePolicy:
    def __init__(self, action_schema="bimanual", action_dim=17):
        self.observation = None
        self.action_schema = action_schema
        self.action_dim = action_dim

    def predict_action(self, observation):
        self.observation = observation
        return {
            "action": torch.zeros(
                (1, 8, self.action_dim), dtype=torch.float32
            )
        }


def make_session():
    policy = FakePolicy()
    value = inference_server.PolicySession(
        policy,
        SimpleNamespace(n_obs_steps=2, n_action_steps=8),
        "cpu",
    )
    return value, policy


class PolicySessionTest(unittest.TestCase):
    def setUp(self):
        self.preprocess = patch.object(
            inference_server,
            "depth_m_to_workspace_cloud",
            side_effect=lambda depth, _k, _rng, device: np.full(
                (4, 3), depth[0, 0], dtype=np.float32
            ),
        )
        self.preprocess.start()
        self.addCleanup(self.preprocess.stop)

    def test_first_observation_is_left_padded(self):
        value, policy = make_session()
        states = np.arange(16, dtype=np.float32)[None]
        depths = np.full((1, 2, 3), 1000, dtype=np.uint16)
        intrinsics = np.eye(3, dtype=np.float64)[None]

        actions, _ = value.infer(states, depths, intrinsics)

        self.assertEqual(actions.shape, (8, 17))
        agent_pos = policy.observation["agent_pos"].numpy()
        point_cloud = policy.observation["point_cloud"].numpy()
        np.testing.assert_array_equal(agent_pos[0, 0], agent_pos[0, 1])
        np.testing.assert_array_equal(point_cloud[0, 0], point_cloud[0, 1])

    def test_explicit_observations_preserve_order(self):
        value, policy = make_session()
        states = np.stack((np.zeros(16, dtype=np.float32),
                           np.ones(16, dtype=np.float32)))
        depths = np.stack((np.full((2, 3), 1000, dtype=np.uint16),
                           np.full((2, 3), 2000, dtype=np.uint16)))
        intrinsics = np.stack((np.eye(3), np.eye(3)))

        value.infer(states, depths, intrinsics)

        agent_pos = policy.observation["agent_pos"].numpy()
        point_cloud = policy.observation["point_cloud"].numpy()
        np.testing.assert_array_equal(agent_pos[0, 0], states[0])
        np.testing.assert_array_equal(agent_pos[0, 1], states[1])
        self.assertAlmostEqual(point_cloud[0, 0, 0, 0], 1.0)
        self.assertAlmostEqual(point_cloud[0, 1, 0, 0], 2.0)

    def test_too_many_observations_are_rejected(self):
        value, _ = make_session()
        with self.assertRaisesRegex(ValueError, "expected 1..2 observations"):
            value.infer(
                np.zeros((3, 16), dtype=np.float32),
                np.zeros((3, 2, 3), dtype=np.uint16),
                np.zeros((3, 3, 3), dtype=np.float64),
            )

    def test_right_only_policy_receives_only_right_state(self):
        policy = FakePolicy("right_only", 9)
        value = inference_server.PolicySession(
            policy,
            SimpleNamespace(n_obs_steps=2, n_action_steps=8),
            "cpu",
        )
        states = np.stack((np.arange(16), np.arange(16) + 20)).astype(
            np.float32
        )
        depths = np.ones((2, 2, 3), dtype=np.uint16)
        intrinsics = np.stack((np.eye(3), np.eye(3)))

        actions, _ = value.infer(states, depths, intrinsics)

        self.assertEqual(actions.shape, (8, 9))
        np.testing.assert_array_equal(
            policy.observation["agent_pos"].numpy()[0], states[:, 8:16]
        )


class DeltaDecodeTest(unittest.TestCase):
    def test_right_only_delta_adds_only_right_arm_anchor(self):
        policy = object.__new__(DP3)
        policy.action_schema = "right_only"
        policy.action_dim = 9
        delta = torch.zeros((1, 4, 9), dtype=torch.float32)
        delta[:, :, 7] = 0.25
        delta[:, :, 8] = 1.0
        states = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

        action = policy._decode_delta_action(
            delta, {"agent_pos": states}, observation_steps=4
        )

        expected_arm = states[:, 3, 0:7].repeat(4, 1)
        torch.testing.assert_close(action[0, :, 0:7], expected_arm)
        torch.testing.assert_close(action[0, :, 7], delta[0, :, 7])
        torch.testing.assert_close(action[0, :, 8], delta[0, :, 8])


if __name__ == "__main__":
    unittest.main()
