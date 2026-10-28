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


class TemporalNearestNeighborBehaviorCloner(NearestNeighborBehaviorCloner):
    """Nearest-neighbor imitation over a short, ordered observation history.

    ``fit_episodes`` keeps episode boundaries explicit so history windows never
    leak across resets. At inference, call ``reset`` before each episode and
    then ``predict`` once per settled macro observation.
    """

    def __init__(self, neighbors: int = 3, stride: int = 8, history: int = 3):
        super().__init__(neighbors=neighbors, stride=stride)
        if history < 2:
            raise ValueError("temporal history must be at least two frames")
        self.history = int(history)
        self._frames: list[np.ndarray] = []

    def _window(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("history window cannot be empty")
        padded = [frames[0]] * max(0, self.history - len(frames)) + frames[-self.history:]
        return np.concatenate([self._feature(frame) for frame in padded])

    def fit_episodes(self, episodes) -> "TemporalNearestNeighborBehaviorCloner":
        features: list[np.ndarray] = []
        labels: list[int] = []
        for observations, actions in episodes:
            if len(observations) != len(actions):
                raise ValueError("each episode must have equally sized observations and actions")
            frames: list[np.ndarray] = []
            for observation, action in zip(observations, actions):
                frames.append(np.asarray(observation))
                features.append(self._window(frames))
                labels.append(int(action))
        if not features:
            raise ValueError("provide at least one non-empty episode")
        self._features = np.stack(features)
        self._labels = np.asarray(labels, dtype=np.int64)
        if np.any((self._labels < 0) | (self._labels >= len(NAVIGATION_ACTIONS))):
            raise ValueError("demonstration action outside navigation action space")
        self.reset()
        return self

    def reset(self) -> None:
        self._frames = []

    def predict(self, observation: np.ndarray) -> int:
        if self._features is None or self._labels is None:
            raise RuntimeError("fit the temporal behavior cloner before prediction")
        self._frames.append(np.asarray(observation))
        self._frames = self._frames[-self.history:]
        query = self._window(self._frames)
        if query.shape[0] != self._features.shape[1]:
            raise ValueError("observation shape differs from demonstrations")
        distances = np.mean((self._features - query) ** 2, axis=1)
        indices = np.argsort(distances, kind="stable")[:min(self.neighbors, len(distances))]
        counts = np.bincount(self._labels[indices], minlength=len(NAVIGATION_ACTIONS))
        return int(np.flatnonzero(counts == counts.max())[0])


class LinearBehaviorCloner:
    """Small balanced multinomial linear classifier for supervised actions.

    Uses fixed-step full-batch gradient descent and no hidden game state. Class
    weighting prevents the common all-UP demonstrator action from dominating
    the less frequent turn labels. This remains imitation learning, not RL.
    """

    def __init__(self, stride: int = 8, epochs: int = 300, learning_rate: float = 0.5,
                 l2: float = 1e-3, seed: int = 0):
        if stride < 1 or epochs < 1 or learning_rate <= 0 or l2 < 0:
            raise ValueError("invalid linear behavior-cloner hyperparameters")
        self.stride = int(stride)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def _feature(self, observation: np.ndarray) -> np.ndarray:
        frame = np.asarray(observation)
        if frame.ndim != 3:
            raise ValueError(f"expected HxWxC observation, got {frame.shape}")
        return frame[::self.stride, ::self.stride, :].astype(np.float32).reshape(-1) / 255.0

    def fit(self, observations, actions) -> "LinearBehaviorCloner":
        x = np.stack([self._feature(item) for item in observations])
        y = np.asarray(actions, dtype=np.int64)
        if not len(y) or len(x) != len(y):
            raise ValueError("provide equally sized, non-empty observations and actions")
        if np.any((y < 0) | (y >= len(NAVIGATION_ACTIONS))):
            raise ValueError("demonstration action outside navigation action space")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-4] = 1.0
        x = np.clip((x - self.mean) / self.scale, -5.0, 5.0)
        counts = np.bincount(y, minlength=len(NAVIGATION_ACTIONS)).astype(np.float32)
        present = counts > 0
        class_weights = np.zeros_like(counts)
        class_weights[present] = len(y) / (present.sum() * counts[present])
        sample_weights = class_weights[y]
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.005, (x.shape[1], len(NAVIGATION_ACTIONS))).astype(np.float32)
        self.bias = np.zeros(len(NAVIGATION_ACTIONS), dtype=np.float32)
        targets = np.eye(len(NAVIGATION_ACTIONS), dtype=np.float32)[y]
        normalizer = max(float(sample_weights.sum()), 1e-8)
        for _ in range(self.epochs):
            logits = x @ self.weights + self.bias
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            errors = (probs - targets) * sample_weights[:, None] / normalizer
            self.weights -= self.learning_rate * (x.T @ errors + self.l2 * self.weights)
            self.bias -= self.learning_rate * errors.sum(axis=0)
        return self

    def predict(self, observation: np.ndarray) -> int:
        if self.weights is None or self.bias is None or self.mean is None or self.scale is None:
            raise RuntimeError("fit the linear behavior cloner before prediction")
        x = np.clip((self._feature(observation) - self.mean) / self.scale, -5.0, 5.0)
        return int(np.argmax(x @ self.weights + self.bias))
