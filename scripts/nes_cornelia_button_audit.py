from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasy-Nes"

STATE_PATH = PROJECT_ROOT / "states" / "ff1_nes_cornelia_start.state"
OUT_DIR = PROJECT_ROOT / "videos" / "nes_cornelia_button_audit"

OUT_DIR.mkdir(parents=True, exist_ok=True)

stable_retro.data.Integrations.add_custom_path(str(CUSTOM_INTEGRATIONS))

env = stable_retro.make(
    game=GAME,
    state=stable_retro.State.NONE,
    inttype=stable_retro.data.Integrations.ALL,
    render_mode="rgb_array",
)

reset_result = env.reset()
obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

buttons = list(getattr(env, "buttons", []))
saved_state = STATE_PATH.read_bytes()

print("Action space:", env.action_space)
print("Buttons:", buttons)
print("Loaded state:", STATE_PATH)
print("Loaded state size:", len(saved_state), "bytes")


def empty_action():
    return np.zeros(env.action_space.shape, dtype=np.int8)


def raw_action(index):
    action = empty_action()
    action[index] = 1
    return action


def step(action):
    global obs
    result = env.step(action)
    obs = result[0]
    return obs


def run_frames(count, action=None):
    if action is None:
        action = empty_action()
    for _ in range(count):
        step(action)


def save(name):
    path = OUT_DIR / name
    Image.fromarray(obs).save(path)
    print("Saved:", path)


for index in range(env.action_space.shape[0]):
    label = buttons[index] if index < len(buttons) and buttons[index] else f"index_{index}"

    print()
    print("-" * 60)
    print("Testing index:", index, "label:", label)

    env.em.set_state(saved_state)
    run_frames(10)
    save(f"index_{index:02d}_{label}_before.png")

    run_frames(60, raw_action(index))
    run_frames(60, empty_action())
    save(f"index_{index:02d}_{label}_after.png")

env.close()

print()
print("Cornelia button audit complete.")
print("Open this folder:")
print(OUT_DIR)
