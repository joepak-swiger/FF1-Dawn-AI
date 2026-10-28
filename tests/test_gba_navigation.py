from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from pathlib import Path
from PIL import Image

from ff1_dawn_ai.gba_env import GBA_ACTIONS
from ff1_dawn_ai.gba_navigation import (
    NAVIGATION_ACTIONS,
    NavigationActionEnv,
    NearestNeighborBehaviorCloner,
)
from scripts.gba_navigation_imitation import aggregate_evaluation, assess_imitation, make_classifier


class FakeEnv:
    action_frames = 1
    observation_space = gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)

    def __init__(self):
        self.calls = []
        self.value = 0
        self.closed = False

    def reset(self, **kwargs):
        self.calls.clear()
        return np.zeros((8, 8, 3), dtype=np.uint8), {"reset": True}

    def step(self, action):
        name = GBA_ACTIONS[action]
        self.calls.append(name)
        self.value += 1
        return np.full((8, 8, 3), self.value, dtype=np.uint8), 0.0, False, False, {
            "emulator_frames_executed": 1,
        }

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class NavigationActionTests(unittest.TestCase):
    def test_action_mapping_holds_direction_then_releases(self):
        raw = FakeEnv()
        env = NavigationActionEnv(raw, hold_frames=3, release_frames=2)
        obs, reward, terminated, truncated, info = env.step(NAVIGATION_ACTIONS.index("LEFT"))
        self.assertEqual(raw.calls, ["LEFT", "LEFT", "LEFT", "NOOP", "NOOP"])
        self.assertEqual(env.action_space.n, 5)
        self.assertEqual(info["navigation_action"], "LEFT")
        self.assertEqual(info["macro_emulator_frames"], 5)
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertEqual(obs.shape, (8, 8, 3))

    def test_noop_never_generates_a_directional_button(self):
        raw = FakeEnv()
        env = NavigationActionEnv(raw, hold_frames=2, release_frames=1)
        env.step(0)
        self.assertEqual(raw.calls, ["NOOP", "NOOP", "NOOP"])

    def test_invalid_action_and_invalid_base_duration_fail_early(self):
        with self.assertRaises(ValueError):
            NavigationActionEnv(FakeEnv(), hold_frames=0)
        raw = FakeEnv()
        raw.action_frames = 8
        with self.assertRaises(ValueError):
            NavigationActionEnv(raw)
        env = NavigationActionEnv(FakeEnv())
        with self.assertRaises(ValueError):
            env.step(5)

    def test_behavior_cloner_learns_demonstrated_labels(self):
        observations = [np.zeros((8, 8, 3), dtype=np.uint8),
                        np.full((8, 8, 3), 255, dtype=np.uint8),
                        np.full((8, 8, 3), 128, dtype=np.uint8)]
        labels = [1, 4, 2]
        policy = NearestNeighborBehaviorCloner(neighbors=1, stride=1).fit(observations, labels)
        self.assertEqual(policy.predict(observations[0]), 1)
        self.assertEqual(policy.predict(observations[1]), 4)
        self.assertEqual(policy.predict(observations[2]), 2)
        with self.assertRaises(ValueError):
            policy.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_reviewed_offset_screens_are_field_anchors(self):
        root = Path(__file__).resolve().parents[1]
        classifier = make_classifier()
        for name in ("center_settled", "west", "east", "south",
                     "west_settled", "east_settled", "south_settled"):
            path = root / "docs/experiments/gba_navigation_imitation" / f"field_anchor_{name}.png"
            with Image.open(path) as source:
                result = classifier.classify(np.asarray(source.convert("RGB")))
            self.assertEqual(result.label, "cornelia_exterior_field", name)
            self.assertEqual(result.status, "candidate", name)
        controls = {
            root / "docs/experiments/gba_scene_classifier/title_screen_negative_control.png": "unknown",
            root / "docs/experiments/gba_castle_goal/20261028_052724_624774Z/town_ram_collision.png": "cornelia_town",
            root / "docs/experiments/gba_castle_goal/20261028_052724_624774Z/party_menu_negative_control.png": "menu",
            root / "docs/experiments/gba_castle_goal/20261028_052724_624774Z/castle_exit_transition.png": "unknown",
        }
        for path, label in controls.items():
            with Image.open(path) as source:
                result = classifier.classify(np.asarray(source.convert("RGB")))
            self.assertEqual(result.label, label, str(path))

    def test_imitation_claim_does_not_overstate_simple_up_holdout(self):
        def row(start, policy, action, success):
            return {"start_id": start, "policy": policy, "goal_reached": success,
                    "action_counts": {name: int(name == action) for name in NAVIGATION_ACTIONS}}
        result = assess_imitation([
            row("west", "BEHAVIOR_CLONING", "RIGHT", True),
            row("west", "ALWAYS_UP", "UP", False),
            row("south", "BEHAVIOR_CLONING", "UP", True),
            row("south", "ALWAYS_UP", "UP", True),
        ], ["west"])
        self.assertTrue(result["training_start_success_with_turns"])
        self.assertEqual(result["held_out_always_up_successes"], 1)
        self.assertFalse(result["held_out_beats_always_up"])
        self.assertIn("no_held_out_turn_challenge", result["claim"])

    def test_imitation_turn_generalization_requires_heldout_up_failure(self):
        def row(start, policy, action, success):
            return {"start_id": start, "policy": policy, "goal_reached": success,
                    "action_counts": {name: int(name == action) for name in NAVIGATION_ACTIONS}}
        result = assess_imitation([
            row("west", "BEHAVIOR_CLONING", "RIGHT", True),
            row("east", "BEHAVIOR_CLONING", "LEFT", True),
            row("east", "ALWAYS_UP", "UP", False),
        ], ["west"])
        self.assertEqual(result["held_out_turn_challenge_ids"], ["east"])
        self.assertEqual(result["held_out_turn_imitation_successes"], 1)
        self.assertTrue(result["held_out_beats_always_up"])

    def test_aggregate_evaluation_keeps_per_start_action_counts(self):
        data = [{"start_id": "west", "policy": "BC", "goal_reached": True,
                 "decision_count": 5, "goal_confirmation_decision": 5,
                 "false_goal_events": 0,
                 "action_counts": {"NOOP": 0, "UP": 4, "DOWN": 0, "LEFT": 0, "RIGHT": 1}}]
        result = aggregate_evaluation(data)
        self.assertEqual(result["BC"]["per_start"]["west"]["action_counts"]["RIGHT"], 1)
        self.assertEqual(result["BC"]["successes"], 1)


if __name__ == "__main__":
    unittest.main()
