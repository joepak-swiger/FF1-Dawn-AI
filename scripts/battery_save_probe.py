from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"
GAME_DIR = CUSTOM_INTEGRATIONS / GAME

SAVE_SOURCE = PROJECT_ROOT / "states" / "ff1_after_party.sav"
OUT_DIR = PROJECT_ROOT / "videos" / "battery_save_probe"

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

if not SAVE_SOURCE.exists():
    raise SystemExit(f"Missing save file: {SAVE_SOURCE}")

print("Battery save source:", SAVE_SOURCE)
print("Battery save size:", SAVE_SOURCE.stat().st_size, "bytes")

# Try the common names mGBA/libretro-style tools may look for.
for name in ["rom.sav", "rom.srm", "FinalFantasyDawnOfSouls.sav", "FinalFantasyDawnOfSouls.srm"]:
    target = GAME_DIR / name
    shutil.copy2(SAVE_SOURCE, target)
    print("Copied battery save to:", target)

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
    global obs
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
    return obs


def run_frames(frame_count, *button_names):
    action = make_action(*button_names) if button_names else np.zeros(env.action_space.shape, dtype=np.int8)
    for _ in range(frame_count):
        step_action(action)


def tap(button_name, hold_frames=10, release_frames=20):
    run_frames(hold_frames, button_name)
    run_frames(release_frames)


def save(name):
    Image.fromarray(obs).save(OUT_DIR / name)


save("000_reset.png")

run_frames(360)
save("001_after_wait.png")

# Step through the early menus slowly, saving each checkpoint.
for i in range(1, 16):
    tap("START")
    save(f"{i:03d}_after_start.png")

    tap("A")
    save(f"{i:03d}_after_a.png")

# From wherever we landed, try common load-game style movements.
branches = {
    "load_try_001_a": ["A", 180],
    "load_try_002_down_a": ["DOWN", "A", 180],
    "load_try_003_start": ["START", 180],
    "load_try_004_down_start": ["DOWN", "START", 180],
    "load_try_005_up_a": ["UP", "A", 180],
}

branch_state = env.em.get_state()

for name, sequence in branches.items():
    env.em.set_state(branch_state)
    run_frames(1)

    for item in sequence:
        if isinstance(item, int):
            run_frames(item)
        else:
            tap(item)

    save(f"{name}.png")

env.close()

print("Battery save probe complete.")
print("Saved screenshots to:")
print(OUT_DIR)
for path in sorted(OUT_DIR.glob("*.png")):
    print(" -", path)
