from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ff1_dawn_ai.nes_env import FF1NesEnv


TOTAL_TIMESTEPS = 8_000

OUT_DIR = PROJECT_ROOT / "videos" / "nes_train_ppo_v0_3_cnn"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "nes_train_ppo_v0_3_cnn"

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


def prepare_obs_for_model(model, obs):
    """
    Raw env gives HWC: 84 x 84 x 3.
    CnnPolicy often expects CHW: 3 x 84 x 84 after VecTransposeImage.
    This makes eval work either way.
    """
    obs = np.asarray(obs)
    policy_shape = model.policy.observation_space.shape

    if obs.shape == policy_shape:
        return obs

    if (
        len(obs.shape) == 3
        and len(policy_shape) == 3
        and obs.shape[0] == policy_shape[1]
        and obs.shape[1] == policy_shape[2]
        and obs.shape[2] == policy_shape[0]
    ):
        return np.transpose(obs, (2, 0, 1))

    return obs


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
    action_counts = {name: 0 for name in env.action_names}

    for step in range(500):
        model_obs = prepare_obs_for_model(model, obs)
        action, _state = model.predict(model_obs, deterministic=deterministic)

        action_int = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = env.step(action_int)

        total_reward += reward
        max_unique_screens = max(max_unique_screens, info["unique_screens"])
        max_same_action_streak = max(max_same_action_streak, info["same_action_streak"])
        max_same_screen_streak = max(max_same_screen_streak, info["same_screen_streak"])
        action_counts[info["button"]] += 1

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

    summary = {
        "label": label,
        "deterministic": deterministic,
        "total_reward": round(total_reward, 4),
        "max_unique_screens": max_unique_screens,
        "max_same_action_streak": max_same_action_streak,
        "max_same_screen_streak": max_same_screen_streak,
        "action_counts": action_counts,
        "gif": str(gif_path),
    }

    print()
    print(f"{label} eval complete.")
    print("Total reward:", summary["total_reward"])
    print("Max unique screens:", max_unique_screens)
    print("Max same action streak:", max_same_action_streak)
    print("Max same screen streak:", max_same_screen_streak)
    print("Action counts:", action_counts)

    return summary


print()
print("FF1 NES PPO v0.3 CNN training run")
print("---------------------------------")
print(f"Training timesteps: {TOTAL_TIMESTEPS}")
print("Policy: CnnPolicy")
print("Note: CNN training can be slower on CPU, so this first run is small.")
print()

vec_env = DummyVecEnv([lambda: make_env(print_buttons=True)])

checkpoint_callback = CheckpointCallback(
    save_freq=2_000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ff1_nes_ppo_v0_3_cnn_checkpoint",
)

model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    n_steps=256,
    batch_size=64,
    n_epochs=2,
    learning_rate=0.00025,
    gamma=0.99,
    ent_coef=0.03,
    device="cpu",
)

print()
print("Training PPO v0.3 CNN...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
)

model_path = MODEL_DIR / "ff1_nes_ppo_v0_3_cnn"
model.save(model_path)
print("Saved final model:", model_path.with_suffix(".zip"))

vec_env.close()

summaries = []
summaries.append(run_eval(model, deterministic=False, label="sampling"))
summaries.append(run_eval(model, deterministic=True, label="deterministic"))

metrics_path = OUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(summaries, indent=2))
print("Saved metrics:", metrics_path)

print()
print("v0.3 CNN training run complete.")
print("Open this folder:")
print(OUT_DIR)
print()
print("Model saved at:")
print(model_path.with_suffix(".zip"))
