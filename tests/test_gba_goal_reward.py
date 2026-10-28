import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from ff1_dawn_ai.gba_goal_reward import FirstCastleEntryReward
from ff1_dawn_ai.gba_landmark import SceneAnchorClassifier


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/experiments/gba_castle_goal/20261028_052724_624774Z"
FIELD = ROOT / "docs/experiments/gba_scene_classifier/cornelia_exterior_checkpoint_anchor.png"
CASTLE = ROOT / "docs/experiments/gba_castle_goal/reference_cornelia_1f.png"


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class FakeFrameEnv(gym.Env):
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(0, 255, (1,), dtype=np.uint8)

    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.array([0], dtype=np.uint8), {}

    def step(self, action):
        self.index = min(self.index + 1, len(self.frames) - 1)
        return np.array([self.index], dtype=np.uint8), 0.0, False, False, {}

    def render(self):
        return self.frames[self.index]


class FirstCastleEntryRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = SceneAnchorClassifier({
            "castle_cornelia_1f": CASTLE,
            "cornelia_exterior_field": EVIDENCE / "overworld_after_castle_exit.png",
            "cornelia_town": EVIDENCE / "town_ram_collision.png",
            "menu": EVIDENCE / "party_menu_negative_control.png",
        }, field_reference_variants=(FIELD,))

    def make_env(self, *, enabled=True, frames=None):
        frames = frames or [rgb(FIELD), rgb(FIELD), rgb(CASTLE), rgb(CASTLE), rgb(CASTLE)]
        return FirstCastleEntryReward(FakeFrameEnv(frames), self.classifier, enabled=enabled)

    def test_only_two_frames_after_confirmed_exterior_reward_and_truncate(self):
        env = self.make_env()
        env.reset()
        self.assertEqual(env.step(0)[1:4], (0.0, False, False))  # confirms field only
        first = env.step(0)
        self.assertFalse(first[4]["first_castle_entry_event"])
        self.assertEqual(first[1], 0.0)
        second = env.step(0)
        self.assertTrue(second[4]["first_castle_entry_event"])
        self.assertEqual(second[1], 1.0)
        self.assertTrue(second[3])
        repeated = env.step(0)
        self.assertFalse(repeated[4]["first_castle_entry_event"])
        self.assertEqual(repeated[1], 0.0)
        env.close()

    def test_default_disabled_is_neutral_and_does_not_truncate(self):
        env = self.make_env(enabled=False)
        env.reset()
        for _ in range(3):
            result = env.step(0)
        self.assertTrue(result[4]["first_castle_entry_event"])
        self.assertEqual(result[1], 0.0)
        self.assertFalse(result[3])
        env.close()

    def test_menu_town_and_nonconsecutive_title_match_never_reward(self):
        frames = [rgb(FIELD), rgb(FIELD), rgb(EVIDENCE / "town_ram_collision.png"),
                  rgb(EVIDENCE / "party_menu_negative_control.png"), rgb(CASTLE)]
        env = self.make_env(frames=frames)
        env.reset()
        rewards = [env.step(0)[1] for _ in range(4)]
        self.assertEqual(rewards, [0.0] * 4)
        env.close()

    def test_reset_clears_gate_and_reward_latch(self):
        env = self.make_env()
        env.reset()
        for _ in range(3):
            env.step(0)
        self.assertTrue(env._rewarded)
        env.reset()
        self.assertFalse(env._rewarded)
        env.close()


if __name__ == "__main__":
    unittest.main()
