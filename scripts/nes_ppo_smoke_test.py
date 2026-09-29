from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ff1_dawn_ai.nes_env import FF1NesEnv


OUT_DIR = PROJECT_ROOT / "videos" / "nes_ppo_smoke_test"
MODEL_DIR = PROJECT_ROOT / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_env():
    return FF1NesEnv(
        project_root=PROJECT_ROOT,
        max_steps=400,
        action_freq=16,
    )


print()
print("FF1 NES PPO smoke test")
print("----------------------")
print("Goal: prove PPO can connect to our FF1 environment without crashing.")
print("This will not make the AI smart yet.")
print()

vec_env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=128,
    batch_size=64,
    n_epochs=1,
    learning_rate=0.00025,
    gamma=0.99,
    device="cpu",
)

print()
print("Training tiny PPO model...")
model.learn(total_timesteps=1024)

model_path = MODEL_DIR / "ff1_nes_ppo_smoke"
model.save(model_path)
print("Saved model:", model_path.with_suffix(".zip"))

vec_env.close()

print()
print("Running trained model rollout...")

env = make_env()
obs, info = env.reset()

frames = []
total_reward = 0.0

for step in range(400):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if step % 5 == 0:
        rgb = env.render()
        frames.append(Image.fromarray(rgb))
        Image.fromarray(rgb).save(OUT_DIR / f"ppo_frame_{step:04d}.png")

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

gif_path = OUT_DIR / "ppo_smoke_rollout.gif"

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
print("PPO smoke test complete.")
print("Total reward:", round(total_reward, 2))
print("Open this folder:")
print(OUT_DIR)
