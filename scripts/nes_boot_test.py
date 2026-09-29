from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasy-Nes"
OUT_DIR = PROJECT_ROOT / "videos" / "nes_boot_test"

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

print("Game:", GAME)
print("Action space:", env.action_space)
print("Observation shape:", obs.shape)
print("Buttons:", getattr(env, "buttons", None))

def save(name):
    path = OUT_DIR / name
    Image.fromarray(obs).save(path)
    print("Saved:", path)

def empty_action():
    return np.zeros(env.action_space.shape, dtype=np.int8)

def button_action(button_name):
    action = empty_action()
    buttons = list(getattr(env, "buttons", []))

    if button_name not in buttons:
        print(f"Button not found: {button_name}")
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

save("000_reset.png")

run_frames(180)
save("001_wait.png")

for i in range(1, 6):
    run_frames(20, button_action("START"))
    run_frames(40)
    save(f"start_press_{i:02d}.png")

for i in range(1, 6):
    run_frames(20, button_action("A"))
    run_frames(40)
    save(f"a_press_{i:02d}.png")

env.close()

print()
print("NES boot test complete.")
print("Open this folder:")
print(OUT_DIR)
