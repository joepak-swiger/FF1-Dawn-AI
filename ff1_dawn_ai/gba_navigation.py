"""Navigation-focused GBA controls and a small behavior-cloning baseline.

These utilities are opt-in layers over ``FF1GbaEnv``. The full eleven-button
environment remains available for menus and battles; navigation experiments
use four cardinal movement macros and NOOP only.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from ff1_dawn_ai.gba_env import GBA_ACTIONS


NAVIGATION_ACTIONS = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT")


class NavigationActionEnv(gym.Env):
    """Expose five movement actions as held-frame macros with a release gap.

    The wrapped environment must use one emulator frame per base step. Each
    navigation decision holds a direction for ``hold_frames``, then sends
    ``NOOP`` for ``release_frames`` so reversal controls do not inherit a
    continuously held button. Rewards from the base environment are passed
    through unchanged (the standard GBA environment returns zero).
    """

    def __init__(self, env, *, hold_frames: int = 8, release_frames: int = 2,
                 frame_observer=None):
        super().__init__()
        if hold_frames < 1 or release_frames < 0:
            raise ValueError("hold_frames must be positive and release_frames non-negative")
        if getattr(env, "action_frames", None) != 1:
            raise ValueError("NavigationActionEnv requires base action_frames=1")
        self.env = env
        self.hold_frames = int(hold_frames)
        self.release_frames = int(release_frames)
        self.frame_observer = frame_observer
        self.action_space = gym.spaces.Discrete(len(NAVIGATION_ACTIONS))
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"navigation action must be in [0, {len(NAVIGATION_ACTIONS) - 1}]")
        name = NAVIGATION_ACTIONS[int(action)]
        obs = None
        reward_total = 0.0
        terminated = truncated = False
        frames = 0
        base_steps = 0
        for control, count in ((name, self.hold_frames), ("NOOP", self.release_frames)):
            base_action = GBA_ACTIONS.index(control)
            for _ in range(count):
                obs, reward, terminated, truncated, info = self.env.step(base_action)
                if self.frame_observer is not None:
                    self.frame_observer(self.env.render())
                reward_total += float(reward)
                frames += int(info.get("emulator_frames_executed", 1))
                base_steps += 1
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        info = dict(info)
        info.update({"navigation_action": name, "held_frames": self.hold_frames if name != "NOOP" else 0,
                     "release_frames": self.release_frames if name != "NOOP" else self.hold_frames + self.release_frames,
                     "macro_base_steps": base_steps, "macro_emulator_frames": frames,
                     "navigation_action_names": NAVIGATION_ACTIONS})
        return obs, reward_total, bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class NearestNeighborBehaviorCloner:
    """Non-parametric supervised imitation over saved observation/action pairs.

    This is deliberately a transparent baseline, not RL. The policy selects
    the majority demonstrated action among the closest training observations.
    Similarity is evaluated only on the low-resolution observation tensor;
    it contains no route index, episode ID, or hidden coordinate.
    """

    def __init__(self, neighbors: int = 3, stride: int = 4):
        if neighbors < 1 or stride < 1:
            raise ValueError("neighbors and stride must be positive")
        self.neighbors = int(neighbors)
        self.stride = int(stride)
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def _feature(self, observation: np.ndarray) -> np.ndarray:
        frame = np.asarray(observation)
        if frame.ndim != 3:
            raise ValueError(f"expected HxWxC observation, got {frame.shape}")
        return frame[::self.stride, ::self.stride, :].astype(np.float32).reshape(-1) / 255.0

    def fit(self, observations, actions) -> "NearestNeighborBehaviorCloner":
        obs = list(observations)
        labels = np.asarray(actions, dtype=np.int64)
        if len(obs) != len(labels) or not len(obs):
            raise ValueError("provide equally sized, non-empty observations and actions")
        self._features = np.stack([self._feature(item) for item in obs])
        if np.any((labels < 0) | (labels >= len(NAVIGATION_ACTIONS))):
            raise ValueError("demonstration action outside navigation action space")
        self._labels = labels
        return self

    def predict(self, observation: np.ndarray) -> int:
        if self._features is None or self._labels is None:
            raise RuntimeError("fit the behavior cloner before prediction")
        query = self._feature(observation)
        if query.shape[0] != self._features.shape[1]:
            raise ValueError("observation shape differs from demonstrations")
        distances = np.mean((self._features - query) ** 2, axis=1)
        indices = np.argsort(distances, kind="stable")[:min(self.neighbors, len(distances))]
        counts = np.bincount(self._labels[indices], minlength=len(NAVIGATION_ACTIONS))
        return int(np.flatnonzero(counts == counts.max())[0])
