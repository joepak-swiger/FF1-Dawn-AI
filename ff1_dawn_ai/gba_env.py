"""Small, reward-neutral Gymnasium wrapper for a native GBA checkpoint.

The wrapper intentionally reports no progress rewards and does not infer game
state. Each reset loads the supplied Stable-Retro core state before returning
the first observation. Actions are a stable discrete mapping to GBA controls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image


GBA_ACTIONS = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "L", "R")
GAME = "FinalFantasyDawnOfSouls-GbAdvance"


class FF1GbaEnv(gym.Env):
    """Gymnasium environment initialized from a saved native GBA state.

    Observation is a stack of ``frame_stack`` resized grayscale frames in
    ``(height, width, stack)`` order. Rewards are always zero. Core-declared
    termination is passed through; ``max_steps`` ends an episode by
    truncation. The caller owns the checkpoint and should pass the dedicated
    custom-party state explicitly.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        project_root: str | Path | None = None,
        max_steps: int = 2_000,
        action_frames: int = 16,
        obs_size: tuple[int, int] = (84, 84),
        frame_stack: int = 3,
        render_mode: str | None = None,
        retro_factory: Callable | None = None,
    ) -> None:
        super().__init__()
        if max_steps < 1 or action_frames < 1 or frame_stack < 1:
            raise ValueError("max_steps, action_frames, and frame_stack must be positive")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = self.project_root / self.checkpoint_path
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing native GBA checkpoint: {self.checkpoint_path}")
        self._checkpoint = self.checkpoint_path.read_bytes()
        if not self._checkpoint:
            raise ValueError(f"Native GBA checkpoint is empty: {self.checkpoint_path}")
        self.max_steps = int(max_steps)
        self.action_frames = int(action_frames)
        self.obs_width, self.obs_height = map(int, obs_size)
        self.frame_stack = int(frame_stack)
        self.render_mode = render_mode

        if retro_factory is None:
            import stable_retro

            stable_retro.data.Integrations.add_custom_path(str(self.project_root / "custom_integrations"))
            retro_factory = lambda **kwargs: stable_retro.make(
                game=GAME,
                state=stable_retro.State.NONE,
                inttype=stable_retro.data.Integrations.ALL,
                use_restricted_actions=stable_retro.Actions.ALL,
                render_mode="rgb_array",
                **kwargs,
            )
        self.retro_env = retro_factory()
        self.button_names = [str(name).upper() if name else None for name in getattr(self.retro_env, "buttons", [])]
        self.button_indices = {name: i for i, name in enumerate(self.button_names) if name}
        required = set(GBA_ACTIONS) - {"NOOP"}
        missing = sorted(required - self.button_indices.keys())
        if missing:
            self.retro_env.close()
            raise ValueError(f"GBA integration is missing required controls: {missing}")

        self.action_space = spaces.Discrete(len(GBA_ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.obs_height, self.obs_width, self.frame_stack),
            dtype=np.uint8,
        )
        self._rgb: np.ndarray | None = None
        self._frames: np.ndarray | None = None
        self._steps = 0
        self._closed = False

    def _empty_action(self) -> np.ndarray:
        return np.zeros(self.retro_env.action_space.shape, dtype=np.int8)

    def _frame(self, obs: np.ndarray) -> np.ndarray:
        rgb = np.asarray(obs)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"Expected RGB observation, got shape {rgb.shape}")
        self._rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
        gray = np.mean(self._rgb, axis=2).astype(np.uint8)
        resized = Image.fromarray(gray).resize(
            (self.obs_width, self.obs_height), Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.uint8)

    def _observe(self, raw_obs: np.ndarray, *, reset: bool = False) -> np.ndarray:
        frame = self._frame(raw_obs)
        if reset or self._frames is None:
            self._frames = np.repeat(frame[:, :, None], self.frame_stack, axis=2)
        else:
            self._frames = np.roll(self._frames, shift=1, axis=2)
            self._frames[:, :, 0] = frame
        return self._frames.copy()

    def _one_step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        result = self.retro_env.step(action)
        if len(result) == 5:
            obs, _reward, terminated, truncated, _info = result
            done = bool(terminated or truncated)
        else:
            obs, _reward, done, _info = result
            done = bool(done)
        return np.asarray(obs), done

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        result = self.retro_env.reset()
        raw = np.asarray(result[0] if isinstance(result, tuple) else result)
        # First load primes this core's GBA framebuffer. Reloading the same
        # checkpoint then produces an identical observation even on the
        # first episode; this was verified against repeated native restores.
        for _restore in range(2):
            self.retro_env.em.set_state(self._checkpoint)
            for _ in range(2):
                raw, done = self._one_step(self._empty_action())
                if done:
                    raise RuntimeError("GBA core terminated while loading the checkpoint")
        self._steps = 0
        observation = self._observe(raw, reset=True)
        return observation, {"checkpoint_path": str(self.checkpoint_path), "checkpoint_loaded": True}

    def step(self, action: int):
        if self._frames is None:
            raise RuntimeError("Call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer in [0, {self.action_space.n - 1}], got {action!r}")
        action_index = int(action)
        name = GBA_ACTIONS[action_index]
        retro_action = self._empty_action()
        if name != "NOOP":
            retro_action[self.button_indices[name]] = 1

        raw = self._rgb
        core_terminated = False
        for _ in range(self.action_frames):
            raw, done = self._one_step(retro_action)
            core_terminated = core_terminated or done
            if done:
                break
        self._steps += 1
        observation = self._observe(raw)
        truncated = self._steps >= self.max_steps and not core_terminated
        return observation, 0.0, bool(core_terminated), bool(truncated), {
            "action": name,
            "step_count": self._steps,
            "neutral_reward": True,
            "checkpoint_loaded_on_reset": True,
        }

    def render(self) -> np.ndarray | None:
        return None if self._rgb is None else self._rgb.copy()

    def close(self) -> None:
        if not self._closed:
            self.retro_env.close()
            self._closed = True
