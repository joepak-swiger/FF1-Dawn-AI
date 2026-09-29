from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ff1_dawn_ai.nes_env import FF1NesEnv


TOTAL_TIMESTEPS = 15_000

OUT_DIR = PROJECT_ROOT / "videos" / "nes_train_ppo_v0_2"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "nes_train_ppo_v0_2"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(print_buttons=False):
    return FF1NesEnv(
        project_root=PROJECT_ROOT,
        max_steps=500,
        action_freq=16,
        print_buttons=print_buttons,
    )


def run_eval(model, deterministic, label):
    print()
    print(f"Running {label} evaluation rollout...")

    env = make_env(print_buttons=True)
    obs, info = env.reset()

    frames = []
    total_reward = 0.0
    max_unique_screens = 0
    max_same_action_streak = 0
    max_same_screen_streak = 0

    for step in range(500):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        max_unique_screens = max(max_unique_screens, info["unique_screens"])
        max_same_action_streak = max(max_same_action_streak, info["same_action_streak"])
        max_same_screen_streak = max(max_same_screen_streak, info["same_screen_streak"])

        if step % 5 == 0:
            rgb = env.render()
            frames.append(Image.fromarray(rgb))
            Image.fromarray(rgb).save(OUT_DIR / f"{label}_frame_{step:04d}.png")

        if step % 25 == 0:
            print(
                f"step={step:04d}",
                f"action={info['button']}",
                f"reward={reward:.2f}",
                f"total={total_reward:.2f}",
                f"unique={info['unique_screens']}",
                f"same_action={info['same_action_streak']}",
                f"same_screen={info['same_screen_streak']}",
            )

        if terminated or truncated:
            print("Episode ended:", "terminated" if terminated else "truncated")
            break

    gif_path = OUT_DIR / f"{label}_eval.gif"

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
    print(f"{label} eval complete.")
    print("Total reward:", round(total_reward, 2))
    print("Max unique screens:", max_unique_screens)
    print("Max same action streak:", max_same_action_streak)
    print("Max same screen streak:", max_same_screen_streak)


print()
print("FF1 NES PPO v0.2 training run")
print("-----------------------------")
print(f"Training timesteps: {TOTAL_TIMESTEPS}")
print("Reward changes: anti-stuck + anti-repeat shaping.")
print()

vec_env = DummyVecEnv([lambda: make_env(print_buttons=True)])

checkpoint_callback = CheckpointCallback(
    save_freq=3_000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ff1_nes_ppo_v0_2_checkpoint",
)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=256,
    batch_size=64,
    n_epochs=2,
    learning_rate=0.00025,
    gamma=0.99,
    ent_coef=0.02,
    device="cpu",
)

print()
print("Training PPO v0.2...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
)

model_path = MODEL_DIR / "ff1_nes_ppo_v0_2"
model.save(model_path)
print("Saved final model:", model_path.with_suffix(".zip"))

vec_env.close()

run_eval(model, deterministic=True, label="deterministic")
run_eval(model, deterministic=False, label="sampling")

print()
print("v0.2 training run complete.")
print("Open this folder:")
print(OUT_DIR)
print()
print("Model saved at:")
print(model_path.with_suffix(".zip"))
