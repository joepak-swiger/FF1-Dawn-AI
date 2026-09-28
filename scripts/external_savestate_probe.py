from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"

STATE_DIR = PROJECT_ROOT / "states"
OUT_DIR = PROJECT_ROOT / "videos" / "external_savestate_probe"

CANDIDATES = [
    STATE_DIR / "ff1_after_party.ss0",
    STATE_DIR / "ff1_after_party.ss1",
    STATE_DIR / "roundtrip_test.state",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)

stable_retro.data.Integrations.add_custom_path(str(CUSTOM_INTEGRATIONS))


def save_image(obs, name):
    path = OUT_DIR / name
    Image.fromarray(obs).save(path)
    print("Saved image:", path)


def empty_action(env):
    return np.zeros(env.action_space.shape, dtype=np.int8)


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
    return obs


def run_frames(env, obs, count):
    action = empty_action(env)
    for _ in range(count):
        obs = step_env(env, action)
    return obs


for candidate in CANDIDATES:
    print()
    print("=" * 70)
    print("Testing:", candidate)

    if not candidate.exists():
        print("Missing file, skipping.")
        continue

    print("File size:", candidate.stat().st_size, "bytes")

    env = stable_retro.make(
        game=GAME,
        state=stable_retro.State.NONE,
        inttype=stable_retro.data.Integrations.ALL,
        render_mode="rgb_array",
    )

    try:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        save_image(obs, f"{candidate.stem}_000_before_set_state.png")

        data = candidate.read_bytes()

        try:
            env.em.set_state(data)
            print("set_state accepted:", candidate.name)
        except Exception:
            print("set_state FAILED for:", candidate.name)
            traceback.print_exc()
            continue

        obs = run_frames(env, obs, 120)
        save_image(obs, f"{candidate.stem}_001_after_set_state.png")

        converted_path = STATE_DIR / f"converted_{candidate.stem}.state"
        converted_path.write_bytes(env.em.get_state())
        print("Wrote converted Stable-Retro state:", converted_path)
        print("Converted state size:", converted_path.stat().st_size, "bytes")

    finally:
        env.close()

print()
print("External savestate probe complete.")
print("Open this folder:")
print(OUT_DIR)
