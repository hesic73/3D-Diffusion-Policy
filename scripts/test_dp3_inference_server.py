from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp3_inference_server as inference_server


class FakePolicy:
    def __init__(self):
        self.observation = None

    def predict_action(self, observation):
        self.observation = observation
        return {"action": torch.zeros((1, 8, 17), dtype=torch.float32)}


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


if __name__ == "__main__":
    unittest.main()
