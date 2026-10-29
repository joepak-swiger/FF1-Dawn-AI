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
    TemporalNearestNeighborBehaviorCloner,
    LinearBehaviorCloner,
)
from scripts.gba_navigation_imitation import (
    aggregate_evaluation, assess_imitation, make_classifier,
    ROUTE_SPECS, TRAIN_START_IDS, VALIDATION_START_IDS, FINAL_TEST_START_IDS,
    select_validation_policy,
)
from scripts.gba_route_discovery import candidate_routes, reset_is_valid
from scripts.gba_navigation_fresh_benchmark import (
    FINAL_ROUTES as FRESH_FINAL_ROUTES,
    TRAIN_IDS as FRESH_TRAIN_IDS,
    VALIDATION_ROUTES as FRESH_VALIDATION_ROUTES,
    make_specs as fresh_benchmark_specs,
    validate_split as validate_fresh_split,
)
from scripts.gba_navigation_recovery import opposite_direction


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

    def test_temporal_cloner_keeps_episode_history_ordered_and_resets(self):
        dark = np.zeros((8, 8, 3), dtype=np.uint8)
        light = np.full((8, 8, 3), 255, dtype=np.uint8)
        model = TemporalNearestNeighborBehaviorCloner(neighbors=1, stride=2, history=3)
        model.fit_episodes([([dark, light], [1, 4]), ([light, dark], [2, 3])])
        model.reset()
        self.assertEqual(model.predict(dark), 1)
        self.assertEqual(model.predict(light), 4)
        model.reset()
        self.assertEqual(model.predict(light), 2)

    def test_balanced_linear_cloner_is_reproducible_and_predicts_legal_actions(self):
        rng = np.random.default_rng(42)
        observations = [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(12)]
        actions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4]
        first = LinearBehaviorCloner(stride=2, epochs=40, seed=9).fit(observations, actions)
        second = LinearBehaviorCloner(stride=2, epochs=40, seed=9).fit(observations, actions)
        self.assertEqual(first.predict(observations[-1]), second.predict(observations[-1]))
        self.assertIn(first.predict(observations[0]), range(len(NAVIGATION_ACTIONS)))

    def test_full_base_gba_actions_remain_available(self):
        self.assertEqual(len(GBA_ACTIONS), 11)
        self.assertTrue({"START", "A", "B", "L", "R"}.issubset(set(GBA_ACTIONS)))

    def test_perturbation_uses_opposite_legal_direction(self):
        for name, opposite in (("UP", "DOWN"), ("DOWN", "UP"),
                               ("LEFT", "RIGHT"), ("RIGHT", "LEFT")):
            self.assertEqual(NAVIGATION_ACTIONS[opposite_direction(NAVIGATION_ACTIONS.index(name))], opposite)
        with self.assertRaises(ValueError):
            opposite_direction(NAVIGATION_ACTIONS.index("NOOP"))

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
        for name in ("west_far", "east_far", "south_far", "east_north", "west_north",
                     "east_south", "west_south"):
            path = root / "docs/experiments/gba_navigation_imitation" / f"field_anchor_reviewed_{name}.png"
            with Image.open(path) as source:
                result = classifier.classify(np.asarray(source.convert("RGB")))
            self.assertEqual(result.label, "cornelia_exterior_field", name)
            self.assertEqual(result.status, "candidate", name)
        for name in ("val_east_north_detour", "val_west_north_detour",
                     "test_east_south_detour", "test_west_south_detour"):
            path = root / "docs/experiments/gba_scene_classifier" / f"field_anchor_reviewed_{name}.png"
            with Image.open(path) as source:
                result = classifier.classify(np.asarray(source.convert("RGB")))
            self.assertEqual(result.label, "cornelia_exterior_field", name)
            self.assertEqual(result.status, "candidate", name)
        for name in ("val_east_north", "val_west_north",
                     "test_east_south", "test_west_south"):
            path = root / "docs/experiments/gba_scene_classifier" / f"field_anchor_reviewed_v2_{name}.png"
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

    def test_turn_generalization_claim_is_per_policy_across_two_start_families(self):
        rows = []
        for start, turn in (("east_south", "LEFT"), ("west_south", "RIGHT")):
            for replicate in range(2):
                rows.append({"start_id": start, "split": "final_test",
                             "policy": "BC_3NN_MULTI_START", "goal_reached": True,
                             "action_counts": {name: int(name in ("UP", turn)) for name in NAVIGATION_ACTIONS}})
                rows.append({"start_id": start, "split": "final_test", "policy": "ALWAYS_UP",
                             "goal_reached": False,
                             "action_counts": {name: int(name == "UP") for name in NAVIGATION_ACTIONS}})
        result = assess_imitation(rows, ["center"])
        self.assertEqual(result["best_learned_policy"], "BC_3NN_MULTI_START")
        self.assertEqual(result["held_out_turn_success_ids"], ["east_south", "west_south"])
        self.assertIn("multiple_held_out_turning", result["claim"])

    def test_aggregate_evaluation_keeps_per_start_action_counts(self):
        data = [{"start_id": "west", "policy": "BC", "goal_reached": True,
                 "decision_count": 5, "goal_confirmation_decision": 5,
                 "false_goal_events": 0,
                 "action_counts": {"NOOP": 0, "UP": 4, "DOWN": 0, "LEFT": 0, "RIGHT": 1}}]
        result = aggregate_evaluation(data)
        self.assertEqual(result["BC"]["per_start"]["west"]["action_counts"]["RIGHT"], 1)
        self.assertEqual(result["BC"]["successes"], 1)

    def test_route_split_is_declared_before_and_separates_diagonal_test_families(self):
        self.assertEqual(len(ROUTE_SPECS), 11)
        self.assertTrue(set(TRAIN_START_IDS).isdisjoint(VALIDATION_START_IDS))
        self.assertTrue(set(TRAIN_START_IDS).isdisjoint(FINAL_TEST_START_IDS))
        self.assertTrue(set(VALIDATION_START_IDS).isdisjoint(FINAL_TEST_START_IDS))
        self.assertEqual(ROUTE_SPECS["east_south"][:4], ["RIGHT"] * 4)
        self.assertEqual(ROUTE_SPECS["east_south"][4:], ["DOWN"] * 4)

    def test_replicated_evaluation_reports_success_rate_and_action_totals(self):
        row = {"start_id": "east_south", "policy": "BC", "goal_reached": True,
               "decision_count": 7, "goal_confirmation_decision": 7,
               "false_goal_events": 0,
               "action_counts": {"NOOP": 0, "UP": 4, "DOWN": 0, "LEFT": 3, "RIGHT": 0}}
        result = aggregate_evaluation([row, {**row, "goal_reached": False, "decision_count": 20}])
        start = result["BC"]["per_start"]["east_south"]
        self.assertEqual(start["episodes"], 2)
        self.assertEqual(start["success_rate"], 0.5)
        self.assertEqual(start["action_counts"]["LEFT"], 6)

    def test_model_selection_does_not_pick_a_zero_success_validation_tie(self):
        results = {"BC_A": {"successes": 0}, "BC_B": {"successes": 0}}
        self.assertIsNone(select_validation_policy(results, ["BC_A", "BC_B"], horizon=48))
        results["BC_B"] = {"successes": 1, "mean_success_decisions": 20}
        self.assertEqual(select_validation_policy(results, ["BC_A", "BC_B"]), "BC_B")

    def test_route_discovery_candidates_are_bounded_and_directional(self):
        east = candidate_routes("east_north", horizon=48)
        west = candidate_routes("west_north", horizon=48)
        self.assertEqual(east, candidate_routes("east_north", horizon=48))
        self.assertGreaterEqual(len(east), 40)
        self.assertTrue(all(len(route["route"]) == 48 for route in east + west))
        self.assertTrue(all(set(route["route"]) <= {"UP", "DOWN", "LEFT", "RIGHT"}
                            for route in east + west))
        self.assertTrue(any(route["segments"][:2] == [["LEFT", 4], ["DOWN", 4]]
                            for route in east))
        self.assertTrue(any(route["segments"][:2] == [["RIGHT", 4], ["DOWN", 4]]
                            for route in west))

    def test_route_reset_gate_requires_repeatability_and_scene_when_requested(self):
        info = {"checkpoint_loaded": True}
        self.assertTrue(reset_is_valid(info, "cornelia_exterior_field", True))
        self.assertFalse(reset_is_valid(info, "unknown", True))
        self.assertFalse(reset_is_valid(info, "cornelia_exterior_field", False))
        self.assertTrue(reset_is_valid(info, "unknown", True, require_field=False))
        self.assertFalse(reset_is_valid(info, "menu", True, require_field=False))

    def test_fresh_benchmark_splits_are_disjoint_and_use_new_four_leg_families(self):
        specs = fresh_benchmark_specs()
        validate_fresh_split(specs)
        self.assertEqual(set(specs), set(FRESH_TRAIN_IDS) |
                         set(FRESH_VALIDATION_ROUTES) | set(FRESH_FINAL_ROUTES))
        self.assertTrue(all(len(route) == 12 for route in
                            list(FRESH_VALIDATION_ROUTES.values()) + list(FRESH_FINAL_ROUTES.values())))
        self.assertTrue(all(len(set(route)) == 4 for route in
                            list(FRESH_VALIDATION_ROUTES.values()) + list(FRESH_FINAL_ROUTES.values())))
        with self.assertRaises(ValueError):
            validate_fresh_split({**specs, "unexpected": ["UP"]})


if __name__ == "__main__":
    unittest.main()
