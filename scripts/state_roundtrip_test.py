from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"

OUT_DIR = PROJECT_ROOT / "videos" / "state_roundtrip_test"
STATE_DIR = PROJECT_ROOT / "states"
STATE_PATH = STATE_DIR / "roundtrip_test.state"

BUTTONS = {
    "B": 0,
    "SELECT": 2,
    "START": 3,
    "UP": 4,
    "DOWN": 5,
    "LEFT": 6,
    "RIGHT": 7,
    "A": 8,
    "L": 10,
    "R": 11,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

stable_retro.data.Integrations.add_custom_path(str(CUSTOM_INTEGRATIONS))

env = stable_retro.make(
    game=GAME,
    state=stable_retro.State.NONE,
    inttype=stable_retro.data.Integrations.ALL,
    render_mode="rgb_array",
)

reset_result = env.reset()
obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result


def make_action(*button_names):
    action = np.zeros(env.action_space.shape, dtype=np.int8)
    for name in button_names:
        action[BUTTONS[name]] = 1
    return action


def step_action(action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    return obs, done


def run_frames(frame_count, *button_names):
    global obs
    action = make_action(*button_names)
    done = False

    for _ in range(frame_count):
        obs, done = step_action(action)
        if done:
            break

    return done


def tap(button_name, hold_frames=8, release_frames=8):
    done = run_frames(hold_frames, button_name)
    if done:
        return True
    return run_frames(release_frames)


def save_image(name):
    Image.fromarray(obs).save(OUT_DIR / name)


# Get to the title screen area.
run_frames(360)
save_image("001_before_saving_state.png")

# Save exact emulator state.
saved_state = env.em.get_state()
print("Saved state type:", type(saved_state))
print("Saved state bytes:", len(saved_state))

STATE_PATH.write_bytes(saved_state)
print("State written to:", STATE_PATH)

# Now change the game by pressing START/A several times.
for _ in range(8):
    tap("START")
    tap("A")

save_image("002_after_changing_screen.png")

# Reload the saved state.
loaded_state = STATE_PATH.read_bytes()
env.em.set_state(loaded_state)

# Step one frame so our obs updates after restoring.
run_frames(1)
save_image("003_after_reloading_state.png")

env.close()

print("Round-trip state test complete.")
print("Saved screenshots to:")
print(OUT_DIR)
for path in sorted(OUT_DIR.glob("*.png")):
    print(" -", path)
