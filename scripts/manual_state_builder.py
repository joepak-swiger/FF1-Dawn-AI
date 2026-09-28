from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"

OUT_DIR = PROJECT_ROOT / "videos" / "manual_state_builder"
STATE_DIR = PROJECT_ROOT / "states"
STATE_PATH = STATE_DIR / "ff1_manual_start.state"

BUTTONS = {
    "b": 0,
    "select": 2,
    "start": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
    "a": 8,
    "l": 10,
    "r": 11,
}

ALIASES = {
    "s": "start",
    "sel": "select",
    "u": "up",
    "d": "down",
    "l": "left",
    "r": "right",
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
step_number = 0


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


def tap(button_name, hold_frames=8, release_frames=18):
    done = run_frames(hold_frames, button_name)
    if done:
        return True
    return run_frames(release_frames)


def save_image(label):
    global step_number
    safe_label = label.replace(" ", "_").replace("/", "_")
    path = OUT_DIR / f"{step_number:04d}_{safe_label}.png"
    current = OUT_DIR / "current.png"

    Image.fromarray(obs).save(path)
    Image.fromarray(obs).save(current)

    print(f"Screen saved: {path}")
    print(f"Current screen: {current}")
    step_number += 1


def macro_auto_party_menu():
    # Known route from boot to the FF1 party setup screen.
    run_frames(360)
    for _ in range(20):
        tap("start")
        tap("a")
    run_frames(60)


def macro_name_all():
    # Auto-name each of the four party members.
    tap("select")
    tap("down")
    tap("select")
    tap("down")
    tap("select")
    tap("down")
    tap("select")
    run_frames(60)


def handle_token(token):
    token = token.strip().lower()
    token = ALIASES.get(token, token)

    if not token:
        return True

    if token in BUTTONS:
        tap(token)
        save_image(token)
        return True

    if token.startswith("wait"):
        parts = token.split(":")
        frames = int(parts[1]) if len(parts) == 2 else 60
        run_frames(frames)
        save_image(f"wait_{frames}")
        return True

    if token == "auto":
        macro_auto_party_menu()
        save_image("auto_party_menu")
        return True

    if token == "nameall":
        macro_name_all()
        save_image("nameall")
        return True

    if token == "save":
        STATE_PATH.write_bytes(env.em.get_state())
        save_image("saved_state")
        print(f"Saved Stable-Retro state to: {STATE_PATH}")
        return True

    if token in {"quit", "exit"}:
        return False

    print(f"Unknown command: {token}")
    return True


save_image("reset")

print()
print("Manual State Builder")
print("--------------------")
print("Open this file in Windows after each command:")
print(r"E:\JOEPAK\FF1-Dawn-AI\videos\manual_state_builder\current.png")
print()
print("Commands:")
print("  auto      = jump through boot/title into FF1 party setup")
print("  nameall   = auto-name all four party members")
print("  a b start select up down left right")
print("  wait or wait:120")
print("  save      = save states/ff1_manual_start.state")
print("  quit")
print()
print("You can type multiple commands on one line, like:")
print("  auto nameall start a")
print()

try:
    while True:
        line = input("buttons> ")
        keep_going = True

        for token in line.split():
            keep_going = handle_token(token)
            if not keep_going:
                break

        if not keep_going:
            break
finally:
    env.close()
    print("Closed emulator.")
