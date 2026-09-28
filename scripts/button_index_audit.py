from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"
OUT_DIR = PROJECT_ROOT / "videos" / "button_index_audit"

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

print("Action space:", env.action_space)
print("env.buttons:", getattr(env, "buttons", None))


def action_for_index(index):
    action = np.zeros(env.action_space.shape, dtype=np.int8)
    action[index] = 1
    return action


def empty_action():
    return np.zeros(env.action_space.shape, dtype=np.int8)


def step(action):
    global obs
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
    return obs


def run_frames(frame_count, action=None):
    if action is None:
        action = empty_action()
    for _ in range(frame_count):
        step(action)


def save(name):
    Image.fromarray(obs).save(OUT_DIR / name)


def tap_index(index, hold_frames=45, release_frames=90):
    run_frames(hold_frames, action_for_index(index))
    save(f"index_{index:02d}_during_hold.png")
    run_frames(release_frames)
    save(f"index_{index:02d}_after_release.png")


# Use our old rough route to reach the setup/menu area.
run_frames(360)

# Mash likely START/A route using the OLD assumptions just to reach the party screen.
# This is only setup; the actual audit below tests every raw index.
for _ in range(20):
    tap_index(3, hold_frames=8, release_frames=8)
    tap_index(8, hold_frames=8, release_frames=8)

run_frames(60)
save("000_before_nameall.png")

# Auto-name using old SELECT/DOWN assumptions.
for index in [2, 5, 2, 5, 2, 5, 2]:
    tap_index(index, hold_frames=8, release_frames=16)

run_frames(90)
save("001_after_nameall_before_audit.png")

named_party_state = env.em.get_state()

# Now the real test: from the exact same state, press each raw input index.
for index in range(env.action_space.shape[0]):
    env.em.set_state(named_party_state)
    run_frames(1)
    save(f"index_{index:02d}_before.png")

    tap_index(index, hold_frames=60, release_frames=180)

env.close()

print("Button index audit complete.")
print("Saved screenshots to:")
print(OUT_DIR)
for path in sorted(OUT_DIR.glob("*.png")):
    print(" -", path)
