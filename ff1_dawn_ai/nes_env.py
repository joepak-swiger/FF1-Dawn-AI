from pathlib import Path
import hashlib

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import stable_retro


class FF1NesEnv(gym.Env):
    """
    PWhiddy-style wrapper for Final Fantasy I NES.

    Starts from our saved Cornelia overworld state.
    Uses a small discrete action space.
    Returns stacked grayscale screen frames.
    Gives reward for seeing new screen states.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        project_root=None,
        max_steps=1000,
        action_freq=16,
        frame_stack=3,
        obs_size=84,
    ):
        super().__init__()

        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.custom_integrations = self.project_root / "custom_integrations"
        self.state_path = self.project_root / "states" / "ff1_nes_cornelia_start.state"

        self.game = "FinalFantasy-Nes"
        self.max_steps = max_steps
        self.action_freq = action_freq
        self.frame_stack = frame_stack
        self.obs_size = obs_size

        stable_retro.data.Integrations.add_custom_path(str(self.custom_integrations))

        self.retro_env = stable_retro.make(
            game=self.game,
            state=stable_retro.State.NONE,
            inttype=stable_retro.data.Integrations.ALL,
            render_mode="rgb_array",
        )

        self.buttons = list(getattr(self.retro_env, "buttons", []))
        print("Stable-Retro buttons:", self.buttons)

        # Movement works on the overworld.
        # A/B are also needed once random battles start.
        self.action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
        self.button_indexes = [4, 5, 6, 7, 8, 0]

        self.action_space = spaces.Discrete(len(self.action_names))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_size, self.obs_size, self.frame_stack),
            dtype=np.uint8,
        )

        self.start_state = self.state_path.read_bytes()
        self.rgb = None
        self.recent_frames = None
        self.seen_hashes = set()
        self.step_count = 0

    def _empty_retro_action(self):
        return np.zeros(self.retro_env.action_space.shape, dtype=np.int8)

    def _retro_action_for_button_index(self, button_index):
        action = self._empty_retro_action()
        action[button_index] = 1
        return action

    def _step_retro(self, action):
        result = self.retro_env.step(action)

        if len(result) == 5:
            self.rgb, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            self.rgb, reward, done, info = result

        return done

    def _run_frames(self, count, action=None):
        if action is None:
            action = self._empty_retro_action()

        done = False
        for _ in range(count):
            done = self._step_retro(action) or done

        return done

    def _press_button_index(self, button_index):
        press_frames = max(2, min(6, self.action_freq // 3))
        release_frames = max(1, self.action_freq - press_frames)

        done = self._run_frames(press_frames, self._retro_action_for_button_index(button_index))
        done = self._run_frames(release_frames, self._empty_retro_action()) or done

        return done

    def _preprocess(self, rgb):
        # RGB -> grayscale -> 84x84
        gray = np.mean(rgb[:, :, :3], axis=2).astype(np.uint8)
        img = Image.fromarray(gray)
        img = img.resize((self.obs_size, self.obs_size), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def _screen_hash(self):
        # Tiny hash so animation noise matters less than full pixels.
        tiny = Image.fromarray(self.rgb).resize((32, 32), Image.Resampling.BILINEAR)
        return hashlib.sha1(np.array(tiny).tobytes()).hexdigest()

    def _get_obs(self):
        frame = self._preprocess(self.rgb)

        if self.recent_frames is None:
            self.recent_frames = np.repeat(frame[:, :, None], self.frame_stack, axis=2)
        else:
            self.recent_frames = np.roll(self.recent_frames, shift=1, axis=2)
            self.recent_frames[:, :, 0] = frame

        return self.recent_frames.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        reset_result = self.retro_env.reset()
        self.rgb = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        self.retro_env.em.set_state(self.start_state)
        self._run_frames(5)

        self.recent_frames = None
        self.seen_hashes = set()
        self.step_count = 0

        screen_key = self._screen_hash()
        self.seen_hashes.add(screen_key)

        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        button_index = self.button_indexes[action]
        button_name = self.action_names[action]

        done = self._press_button_index(button_index)

        screen_key = self._screen_hash()
        is_new_screen = screen_key not in self.seen_hashes

        reward = 1.0 if is_new_screen else -0.01

        if is_new_screen:
            self.seen_hashes.add(screen_key)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = bool(done)

        info = {
            "button": button_name,
            "unique_screens": len(self.seen_hashes),
            "step_count": self.step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return self.rgb

    def close(self):
        self.retro_env.close()
