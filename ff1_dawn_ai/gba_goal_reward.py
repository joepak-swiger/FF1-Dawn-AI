"""Opt-in sparse reward for the first temporally confirmed Castle entry.

The base FF1GbaEnv remains reward-neutral. This wrapper requires an explicit
scene classifier and emits a reward only on the first confirmed transition
from a confirmed Cornelia exterior field to Castle Cornelia 1F.
"""

from __future__ import annotations

import gymnasium as gym

from ff1_dawn_ai.gba_landmark import SceneAnchorClassifier, SceneConfirmationGate


class FirstCastleEntryReward(gym.Wrapper):
    """Optional one-shot sparse goal reward; disabled wrappers remain neutral."""

    def __init__(self, env: gym.Env, classifier: SceneAnchorClassifier, *,
                 enabled: bool = False, reward_value: float = 1.0,
                 terminate_on_goal: bool = True, confirmation_frames: int = 2):
        super().__init__(env)
        if reward_value <= 0:
            raise ValueError("reward_value must be positive")
        self.classifier = classifier
        self.enabled = bool(enabled)
        self.reward_value = float(reward_value)
        self.terminate_on_goal = bool(terminate_on_goal)
        self.gate = SceneConfirmationGate(confirmation_frames)
        self._rewarded = False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.gate.reset()
        self._rewarded = False
        initial = self.classifier.classify(self.env.render())
        update = self.gate.update(initial)
        info = dict(info)
        info.update({"scene_candidate": initial.label, "scene_candidate_status": initial.status,
                     "confirmed_scene": update.confirmed_scene,
                     "castle_entry_reward_enabled": self.enabled})
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        result = self.classifier.classify(self.env.render())
        update = self.gate.update(result)
        goal_event = bool(update.first_castle_entry and not self._rewarded)
        if goal_event:
            self._rewarded = True
        reward = float(base_reward) + (self.reward_value if goal_event and self.enabled else 0.0)
        goal_truncation = goal_event and self.enabled and self.terminate_on_goal
        info = dict(info)
        info.update({"scene_candidate": result.label,
                     "scene_candidate_status": result.status,
                     "confirmed_scene": update.confirmed_scene,
                     "scene_transition": update.transitioned,
                     "first_castle_entry_event": goal_event,
                     "castle_entry_reward": self.reward_value if goal_event and self.enabled else 0.0,
                     "castle_entry_reward_enabled": self.enabled})
        return observation, reward, terminated, bool(truncated or goal_truncation), info
