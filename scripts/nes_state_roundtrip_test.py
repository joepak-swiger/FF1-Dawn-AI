from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasy-Nes"

OUT_DIR = PROJECT_ROOT / "videos" / "nes_state_roundtrip_test"
STATE_DIR = PROJECT_ROOT / "states"
STATE_PATH = STATE_DIR / "ff1_nes_roundtrip_test.state"

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

buttons = list(getattr(env, "buttons", []))
print("Buttons:", buttons)


def empty_action():
    return np.zeros(env.action_space.shape, dtype=np.int8)


def button_action(button_name):
    action = empty_action()

    if button_name not in buttons:
        print("Button not found:", button_name)
        return action

    action[buttons.index(button_name)] = 1
    return action


def step(action):
    global obs
    result = env.step(action)
    obs = result[0]
    return obs


def run_frames(frame_count, action=None):
    if action is None:
        action = empty_action()

    for _ in range(frame_count):
        step(action)


def tap(button_name, hold=15, release=30):
    print("Tap:", button_name)
    run_frames(hold, button_action(button_name))
    run_frames(release, empty_action())


def save_image(name):
    path = OUT_DIR / name
    Image.fromarray(obs).save(path)
    print("Saved image:", path)


run_frames(180)
save_image("000_before_save.png")

saved_state = env.em.get_state()
STATE_PATH.write_bytes(saved_state)

print("Saved native Stable-Retro state:", STATE_PATH)
print("State size:", STATE_PATH.stat().st_size, "bytes")

tap("START")
tap("A")
tap("START")
tap("A")
save_image("001_after_changing_screen.png")

env.em.set_state(saved_state)

immediate_restored_state = env.em.get_state()
print("Immediate restored state size:", len(immediate_restored_state), "bytes")
print("Exact state match immediately after restore:", immediate_restored_state == saved_state)

run_frames(5)
save_image("002_after_restore.png")

restored_state_after_frames = env.em.get_state()

print("Restore test complete.")
print("Restored state size after 5 frames:", len(restored_state_after_frames), "bytes")
print("Exact state match after 5 frames:", restored_state_after_frames == saved_state)
print("Open this folder:")
print(OUT_DIR)

env.close()
