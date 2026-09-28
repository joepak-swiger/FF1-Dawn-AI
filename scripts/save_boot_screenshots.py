from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"
OUT_DIR = PROJECT_ROOT / "videos" / "boot_screenshots"

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

# Save the first frame.
Image.fromarray(obs).save(OUT_DIR / "frame_000_reset.png")

# Press no buttons for a few chunks and save what the screen looks like.
noop = np.zeros(env.action_space.shape, dtype=np.int8)

save_at_steps = {30, 60, 120, 240, 360}

for step in range(1, 361):
    step_result = env.step(noop)

    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        obs, reward, done, info = step_result

    if step in save_at_steps:
        Image.fromarray(obs).save(OUT_DIR / f"frame_{step:03d}.png")

    if done:
        print(f"Episode ended early at step {step}")
        break

env.close()

print("Saved screenshots to:")
print(OUT_DIR)
for path in sorted(OUT_DIR.glob("*.png")):
    print(" -", path)
