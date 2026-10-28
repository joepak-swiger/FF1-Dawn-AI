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

    def test_seeded_direction_policy_is_reproducible_and_covers_directions(self):
        from scripts.gba_gym_smoke import seeded_direction_actions

        actions = seeded_direction_actions(64, seed=1729)
        self.assertEqual(actions, seeded_direction_actions(64, seed=1729))
        self.assertEqual(set(actions), {"UP", "DOWN", "LEFT", "RIGHT"})

    def test_checkpoint_discovery_requires_exact_seed_roster(self):
        from unittest.mock import patch
        from scripts import gba_gym_smoke

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for run_id, last_name in (("older-seed", "Seed"), ("newer-mustar", "Mustard")):
                report = root / "videos" / "gba_feasibility" / run_id
                report.mkdir(parents=True)
                names = ["Sif", "Myth", "Milo", last_name]
                metrics = {"runtime": {
                    "gameplay_readiness": "verified",
                    "party_state_machine": {
                        "entered_names": [
                            {"canonical_name": name, "display_name": name} for name in names
                        ],
                        "classes": ["Warrior", "Thief", "Monk", "Red Mage"],
                    },
                }}
                (report / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                state = root / "states" / "gba_custom_party" / run_id / "stable_retro_core.state"
                state.parent.mkdir(parents=True)
                state.write_bytes(run_id.encode())
            with patch.object(gba_gym_smoke, "ROOT", root):
                self.assertEqual(gba_gym_smoke.discover_checkpoint(),
                                 root / "states/gba_custom_party/older-seed/stable_retro_core.state")

    def test_identical_observation_streak_is_pixel_diagnostic(self):
        from scripts.gba_gym_smoke import longest_repeat_streak

        self.assertEqual(longest_repeat_streak(["a", "a", "b", "b", "b", "c"]), 3)
        self.assertEqual(longest_repeat_streak([]), 0)

    def test_reset_replays_same_observation_and_preserves_checkpoint(self):
        first, first_info = self.env.reset(seed=11)
        second, second_info = self.env.reset(seed=11)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info["native_restore_count"], 2)
        self.assertEqual(first_info["priming_noop_frames"], 4)
        self.assertEqual(first_info["reset_index"], 1)
        self.assertEqual(second_info["reset_index"], 2)
        self.assertEqual(self.checkpoint.read_bytes(), b"safe native fixture")

    def test_evaluation_baselines_and_visual_metrics_are_explicit_and_deterministic(self):
        from scripts.gba_eval import POLICIES, actions_for, changed_fraction, estimate_scene_translation, marker_candidate, shift_matches_direction

        self.assertIn("NOOP", POLICIES)
        self.assertIn("RANDOM", POLICIES)
        self.assertTrue(set(("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "L", "R")).issubset(POLICIES))
        self.assertEqual(actions_for("UP", 3, 2), ["UP"] * 3)
        self.assertEqual(actions_for("RANDOM", 20, 1729), actions_for("RANDOM", 20, 1729))
        blank = np.zeros((160, 240, 3), dtype=np.uint8)
        changed = blank.copy()
        changed[10, 10] = 255
        self.assertAlmostEqual(changed_fraction(blank, changed), 1 / (160 * 240))
        self.assertIsNone(marker_candidate(blank))
        avatar = blank.copy()
        avatar[80:83, 110:114] = (220, 110, 30)
        marker = marker_candidate(avatar)
        self.assertEqual(marker["confidence"], "low_unvalidated")
        self.assertEqual(marker["matching_pixels"], 12)

    def test_scene_translation_estimator_recovers_synthetic_screen_shift(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv is optional; scene translation remains unavailable")
        from scripts.gba_eval import estimate_scene_translation, shift_matches_direction

        rng = np.random.default_rng(3)
        frame = rng.integers(0, 255, size=(160, 240, 3), dtype=np.uint8)
        transform = np.float32([[1, 0, 7], [0, 1, -4]])
        shifted = cv2.warpAffine(frame, transform, (240, 160))
        result = estimate_scene_translation(frame, shifted)
        self.assertAlmostEqual(result["dx_pixels_candidate"], 7, delta=0.2)
        self.assertAlmostEqual(result["dy_pixels_candidate"], -4, delta=0.2)
        self.assertTrue(shift_matches_direction("LEFT", {"dx_pixels_candidate": 12, "dy_pixels_candidate": 0}))
        self.assertFalse(shift_matches_direction("UP", {"dx_pixels_candidate": 0, "dy_pixels_candidate": -12}))

    def test_single_restore_configuration_is_available_for_reset_audit(self):
        one_restore = FF1GbaEnv(
            self.checkpoint, max_steps=1, action_frames=1, restore_count=1,
            priming_frames_per_restore=2, retro_factory=lambda: self.stub,
        )
        try:
            first, info = one_restore.reset(seed=1)
            second, later_info = one_restore.reset(seed=1)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(info["native_restore_count"], 1)
            self.assertEqual(info["priming_noop_frames"], 2)
            self.assertEqual(later_info["reset_index"], 2)
        finally:
            one_restore.close()

    def test_gba_eval_evidence_packager_strips_raw_checkpoint_metadata_and_fails_closed(self):
        from scripts.package_gba_eval_evidence import package

        source = Path(self.temp.name) / "local-run"
        source.mkdir()
        metrics = {
            "run_id": "run-1", "checkpoint_run_id": "seed-1",
            "checkpoint_sha256_before": "private", "checkpoint_unchanged": True,
            "reset_observation_hashes_identical": True, "reset_count": 4,
            "reset_strategy_audit": {"1": {"first_equals_later": True}},
            "mean_reset_seconds": 0.1,
            "policies": ["NOOP", "UP"], "repeats_per_policy": 2,
            "fixed_horizon_actions": 3, "action_frames": 8,
            "trials_completed": 6, "extra_control_trials": 2,
            "blocked_terrain_probe": {"horizon_actions": 24},
            "action_steps_completed": 60,
            "neutral_rewards_only": True,
            "mean_env_step_seconds": 0.01,
            "episode_summary": {"NOOP": {}, "UP": {}}, "interpretation": {},
        }
        (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        for name in (
            "noop_r1_s001.png", "up_r1_s008.png", "down_r1_s008.png",
            "left_r1_s008.png", "right_r1_s008.png", "start_r1_s008.png",
            "edge_noop_s024.png", "edge_right_s024.png",
        ):
            Image.new("RGB", (30, 20), (25, 80, 130)).save(source / name)
        out = package(source, Path(self.temp.name) / "published")
        public = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        self.assertNotIn("checkpoint_sha256_before", public)
        self.assertTrue((out / "proof.png").is_file())
        self.assertTrue((out / "REPORT.md").is_file())
        metrics["checkpoint_unchanged"] = False
        (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Refusing incomplete"):
            package(source, Path(self.temp.name) / "rejected")

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
            "run_id": "synthetic", "checkpoint_run_id": "seed-run", "checkpoint_loaded": True,
            "reset_reproducible_observation": True, "observation_shape": [84, 84, 3],
            "observation_dtype": "uint8", "action_names": ["NOOP", "UP"],
            "requested_steps": 2, "steps_per_episode": 1, "completed_steps": 2, "action_frames": 4,
            "episodes_requested": 2, "episodes_completed": 2,
            "episodes_reproducible": True, "action_counts": {},
            "longest_identical_observation_streak": 1,
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
