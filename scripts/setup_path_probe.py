from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"
OUT_DIR = PROJECT_ROOT / "videos" / "setup_path_probe"

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


def tap(button_name, hold_frames=8, release_frames=8):
    done = run_frames(hold_frames, button_name)
    if done:
        return True
    return run_frames(release_frames)


def save(name):
    Image.fromarray(obs).save(OUT_DIR / name)


save("000_reset.png")

# Same rough path that got us to the FF1 setup menu earlier.
run_frames(360)
save("001_after_wait.png")

for i in range(1, 21):
    tap("START")
    tap("A")

    if i in {5, 10, 15, 20}:
        save(f"setup_{i:03d}_after_start_a_pairs.png")

# Now test extra inputs from the party setup/menu screen.
extra_steps = [
    ("A",),
    ("A",),
    ("A",),
    ("START",),
    ("A",),
    ("A",),
    ("DOWN",),
    ("A",),
    ("A",),
    ("START",),
    ("A",),
]

for index, buttons in enumerate(extra_steps, start=1):
    for button in buttons:
        tap(button)

    run_frames(30)
    label = "_".join(buttons)
    save(f"extra_{index:03d}_{label}.png")

env.close()

print("Setup path probe complete.")
print("Saved screenshots to:")
print(OUT_DIR)
for path in sorted(OUT_DIR.glob("*.png")):
    print(" -", path)
