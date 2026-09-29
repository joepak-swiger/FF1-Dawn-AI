from pathlib import Path
import random
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ff1_dawn_ai.nes_env import FF1NesEnv


OUT_DIR = PROJECT_ROOT / "videos" / "nes_pwhiddy_style_random_rollout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

env = FF1NesEnv(
    project_root=PROJECT_ROOT,
    max_steps=400,
    action_freq=16,
)

obs, info = env.reset()

frames = []
total_reward = 0.0

print()
print("FF1 NES PWhiddy-style random rollout")
print("------------------------------------")
print("Actions:", env.action_names)
print("Output folder:", OUT_DIR)
print()

for step in range(400):
    action = random.randrange(env.action_space.n)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 5 == 0:
        rgb = env.render()
        frames.append(Image.fromarray(rgb))
        Image.fromarray(rgb).save(OUT_DIR / f"frame_{step:04d}.png")

    if step % 25 == 0:
        print(
            f"step={step:04d}",
            f"action={info['button']}",
            f"reward={reward:.2f}",
            f"total={total_reward:.2f}",
            f"unique_screens={info['unique_screens']}",
        )

    if terminated or truncated:
        print("Episode ended:", "terminated" if terminated else "truncated")
        break

gif_path = OUT_DIR / "random_rollout.gif"

if frames:
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    print("Saved GIF:", gif_path)

env.close()

print()
print("Random rollout complete.")
print("Total reward:", round(total_reward, 2))
print("Open this folder:")
print(OUT_DIR)
