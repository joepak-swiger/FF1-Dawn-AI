from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"
OUT_DIR = PROJECT_ROOT / "videos" / "finish_party_probe_2"

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


def tap(button_name, hold_frames=8, release_frames=12):
    done = run_frames(hold_frames, button_name)
    if done:
        return True
    return run_frames(release_frames)


def save(name):
    Image.fromarray(obs).save(OUT_DIR / name)


def press_sequence(sequence):
    for item in sequence:
        if isinstance(item, int):
            run_frames(item)
        else:
            tap(item)


# Get to the setup area.
run_frames(360)

for _ in range(20):
    tap("START")
    tap("A")

run_frames(60)
save("000_after_basic_setup_route.png")

# Try to force back to the New Game party menu.
# If we are in Select a class, B should back out.
for _ in range(3):
    tap("B")
    run_frames(30)

save("001_after_b_backouts.png")

menu_state = env.em.get_state()

candidates = {
    "001_start_done": ["START", 180],
    "002_start_then_a": ["START", 90, "A", 180],
    "003_select_then_start": ["SELECT", 60, "START", 180],
    "004_select_start_a": ["SELECT", 60, "START", 90, "A", 180],
    "005_auto_name_each_then_start": [
        "SELECT", 40,
        "DOWN", 40,
        "SELECT", 40,
        "DOWN", 40,
        "SELECT", 40,
        "DOWN", 40,
        "SELECT", 40,
        "START", 240,
    ],
    "006_auto_name_each_start_a": [
        "SELECT", 40,
        "DOWN", 40,
        "SELECT", 40,
        "DOWN", 40,
        "SELECT", 40,
        "DOWN", 40,
        "SELECT", 40,
        "START", 90,
        "A", 240,
    ],
    "007_start_confirm_a_mash": [
        "START", 60,
        "A", 60,
        "A", 60,
        "A", 60,
        "A", 240,
    ],
    "008_auto_names_start_a_mash": [
        "SELECT", "DOWN",
        "SELECT", "DOWN",
        "SELECT", "DOWN",
        "SELECT",
        "START",
        "A", "A", "A", "A", "A",
        360,
    ],
}

for name, sequence in candidates.items():
    env.em.set_state(menu_state)
    run_frames(1)
    save(f"{name}_before.png")

    press_sequence(sequence)
    save(f"{name}_after.png")

env.close()

print("Finish party probe 2 complete.")
print("Saved screenshots to:")
print(OUT_DIR)
for path in sorted(OUT_DIR.glob("*.png")):
    print(" -", path)
