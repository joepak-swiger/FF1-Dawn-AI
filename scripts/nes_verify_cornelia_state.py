from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasy-Nes"

STATE_PATH = PROJECT_ROOT / "states" / "ff1_nes_cornelia_start.state"
OUT_DIR = PROJECT_ROOT / "videos" / "nes_verify_cornelia_state"

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
print("Buttons:", buttons)

saved_state = STATE_PATH.read_bytes()
print("Loaded state:", STATE_PATH)
print("Loaded state size:", len(saved_state), "bytes")

env.em.set_state(saved_state)

def empty_action():
    return np.zeros(env.action_space.shape, dtype=np.int8)

def button_action(button_name):
    action = empty_action()
    if button_name in buttons:
        action[buttons.index(button_name)] = 1
    else:
        print("Missing button:", button_name)
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

def tap(button_name, hold=15, release=30):
    print("Tap:", button_name)
    run_frames(hold, button_action(button_name))
    run_frames(release, empty_action())

def save(name):
    path = OUT_DIR / name
    Image.fromarray(obs).save(path)
    print("Saved:", path)

run_frames(5)
save("000_loaded_cornelia_state.png")

tap("A")
save("001_after_A.png")

env.em.set_state(saved_state)
run_frames(5)
tap("START")
save("002_after_START.png")

env.em.set_state(saved_state)
run_frames(5)
tap("SELECT")
save("003_after_SELECT.png")

env.close()

print()
print("Cornelia state verification complete.")
print("Open this folder:")
print(OUT_DIR)
