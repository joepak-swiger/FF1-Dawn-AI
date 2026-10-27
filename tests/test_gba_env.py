import tempfile
import unittest
import json
from pathlib import Path

import gymnasium
import numpy as np
from PIL import Image
from gymnasium.utils.env_checker import check_env

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS
from scripts.gba_gym_smoke import package_public_evidence


class _Emulator:
    def __init__(self, owner):
        self.owner = owner

    def set_state(self, state):
        self.owner.loaded_state = state
        self.owner.tick = 0


class _RetroStub:
    buttons = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", None, "R", "L"]
    action_space = gymnasium.spaces.MultiBinary(len(buttons))

    def __init__(self):
        self.tick = 0
        self.actions_seen = []
        self.loaded_state = None
        self.terminate_at = None
        self.em = _Emulator(self)
        self.closed = False

    def _image(self):
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        frame[:, :, 0] = self.tick % 255
        return frame

    def reset(self):
        self.tick = 0
        return self._image(), {}

    def step(self, action):
        self.actions_seen.append(np.asarray(action).copy())
        self.tick += 1
        done = self.terminate_at is not None and self.tick >= self.terminate_at
        return self._image(), 0.0, done, False, {}

    def close(self):
        self.closed = True


class GbaGymnasiumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "native.state"
        self.checkpoint.write_bytes(b"safe native fixture")
        self.stub = _RetroStub()
        self.env = FF1GbaEnv(
            self.checkpoint,
            max_steps=2,
            action_frames=3,
            obs_size=(84, 84),
            frame_stack=3,
            retro_factory=lambda: self.stub,
        )
        self.addCleanup(self.env.close)

    def test_gymnasium_api_observation_and_checkpoint_restore(self):
        check_env(self.env, skip_render_check=True)
        observation, info = self.env.reset(seed=7)
        self.assertEqual(observation.shape, (84, 84, 3))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertTrue(info["checkpoint_loaded"])
        self.assertEqual(self.stub.loaded_state, b"safe native fixture")
        self.assertEqual(self.env.render().shape, (160, 240, 3))

    def test_discrete_mapping_neutral_reward_and_truncation(self):
        self.env.reset()
        action_index = GBA_ACTIONS.index("RIGHT")
        _, reward, terminated, truncated, info = self.env.step(action_index)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action"], "RIGHT")
        self.assertEqual(np.flatnonzero(self.stub.actions_seen[-1]).tolist(), [self.stub.buttons.index("RIGHT")])
        _, reward, terminated, truncated, _ = self.env.step(GBA_ACTIONS.index("NOOP"))
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(np.flatnonzero(self.stub.actions_seen[-1]).tolist(), [])
        self.assertEqual(len(self.stub.actions_seen), 4 + 3 + 3)

    def test_reset_replays_same_observation_and_preserves_checkpoint(self):
        first, _ = self.env.reset(seed=11)
        second, _ = self.env.reset(seed=11)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(self.checkpoint.read_bytes(), b"safe native fixture")

    def test_core_termination_is_forwarded_without_time_limit_truncation(self):
        self.env.reset()
        self.stub.terminate_at = self.stub.tick + 2
        _, reward, terminated, truncated, _ = self.env.step(GBA_ACTIONS.index("UP"))
        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_invalid_actions_and_missing_checkpoint_fail_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.env.step(0)
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(self.env.action_space.n)
        with self.assertRaises(FileNotFoundError):
            FF1GbaEnv(Path(self.temp.name) / "missing.state", retro_factory=lambda: self.stub)

    def test_public_smoke_package_is_sanitized_and_fails_closed(self):
        source = Path(self.temp.name) / "run"
        source.mkdir()
        metrics = {
            "run_id": "synthetic", "checkpoint_loaded": True,
            "reset_reproducible_observation": True, "observation_shape": [84, 84, 3],
            "observation_dtype": "uint8", "action_names": ["NOOP", "UP"],
            "requested_steps": 1, "completed_steps": 1, "action_frames": 4,
            "neutral_rewards_only": True, "terminated": False, "truncated": True,
            "checkpoint_source_unchanged": True, "checkpoint_sha256": "private-detail-to-strip",
            "absolute_checkpoint_path": "C:/private/path/native.state", "limitations": [],
        }
        (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        Image.new("RGB", (32, 16), (15, 75, 120)).save(source / "contact_sheet.png")
        public = package_public_evidence(source, Path(self.temp.name) / "published")
        published_metrics = json.loads((public / "metrics.json").read_text(encoding="utf-8"))
        self.assertNotIn("checkpoint_sha256", published_metrics)
        self.assertNotIn("absolute_checkpoint_path", published_metrics)
        self.assertTrue((public / "REPORT.md").is_file())
        with self.assertRaises(FileExistsError):
            package_public_evidence(source, public)
        metrics["terminated"] = True
        metrics["truncated"] = False
        (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        terminal_public = package_public_evidence(source, Path(self.temp.name) / "terminal-published")
        self.assertTrue((terminal_public / "REPORT.md").is_file())
        metrics["reset_reproducible_observation"] = False
        (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nondeterministic"):
            package_public_evidence(source, Path(self.temp.name) / "rejected")


if __name__ == "__main__":
    unittest.main()
