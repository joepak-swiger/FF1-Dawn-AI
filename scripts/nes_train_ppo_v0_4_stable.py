from pathlib import Path
import json
import shutil
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ff1_dawn_ai.nes_env import FF1NesEnv


TOTAL_TIMESTEPS = 12_000
EVAL_STEPS = 500
CHECKPOINT_EVAL_STEPS = 300
CHECKPOINT_EVAL_RUNS = 2

OUT_DIR = PROJECT_ROOT / "videos" / "nes_train_ppo_v0_4_stable"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "nes_train_ppo_v0_4_stable"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(print_buttons=False, max_steps=500):
    return FF1NesEnv(
        project_root=PROJECT_ROOT,
        max_steps=max_steps,
        action_freq=16,
        print_buttons=print_buttons,
    )


def prepare_obs_for_model(model, obs):
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


def run_eval(model, label, steps=500, save_frames=True, deterministic=False):
    env = make_env(print_buttons=False, max_steps=steps)
    obs, info = env.reset()

    frames = []
    total_reward = 0.0
    max_unique_screens = 0
    max_same_action_streak = 0
    max_same_screen_streak = 0
    action_counts = {name: 0 for name in env.action_names}

    for step in range(steps):
        model_obs = prepare_obs_for_model(model, obs)
        action, _state = model.predict(model_obs, deterministic=deterministic)

        action_int = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = env.step(action_int)

        total_reward += reward
        max_unique_screens = max(max_unique_screens, info["unique_screens"])
        max_same_action_streak = max(max_same_action_streak, info["same_action_streak"])
        max_same_screen_streak = max(max_same_screen_streak, info["same_screen_streak"])
        action_counts[info["button"]] += 1

        if save_frames and step % 5 == 0:
            rgb = env.render()
            frames.append(Image.fromarray(rgb))
            Image.fromarray(rgb).save(OUT_DIR / f"{label}_frame_{step:04d}.png")

        if save_frames and step % 25 == 0:
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
            break

    gif_path = None

    if save_frames and frames:
        gif_path = OUT_DIR / f"{label}_eval.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
        )
        print("Saved GIF:", gif_path)

    env.close()

    return {
        "label": label,
        "deterministic": deterministic,
        "total_reward": round(total_reward, 4),
        "max_unique_screens": max_unique_screens,
        "max_same_action_streak": max_same_action_streak,
        "max_same_screen_streak": max_same_screen_streak,
        "action_counts": action_counts,
        "gif": str(gif_path) if gif_path else None,
    }


def score_checkpoint(model_path):
    model = PPO.load(model_path, device="cpu")

    runs = []
    for run_index in range(CHECKPOINT_EVAL_RUNS):
        result = run_eval(
            model,
            label=f"checkpoint_tmp_{model_path.stem}_{run_index}",
            steps=CHECKPOINT_EVAL_STEPS,
            save_frames=False,
            deterministic=False,
        )
        runs.append(result)

    mean_reward = float(np.mean([r["total_reward"] for r in runs]))
    mean_unique = float(np.mean([r["max_unique_screens"] for r in runs]))
    max_repeat = max(r["max_same_action_streak"] for r in runs)

    return {
        "model_path": str(model_path),
        "mean_reward": round(mean_reward, 4),
        "mean_unique_screens": round(mean_unique, 4),
        "max_same_action_streak": max_repeat,
        "runs": runs,
    }


print()
print("FF1 NES PPO v0.4 stable CNN training")
print("------------------------------------")
print(f"Training timesteps: {TOTAL_TIMESTEPS}")
print("Policy: CnnPolicy")
print("Changes: lower learning rate, target_kl, checkpoint comparison.")
print()

vec_env = DummyVecEnv([lambda: make_env(print_buttons=True, max_steps=500)])

checkpoint_callback = CheckpointCallback(
    save_freq=2_000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ff1_nes_ppo_v0_4_checkpoint",
)

model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    n_steps=256,
    batch_size=64,
    n_epochs=1,
    learning_rate=0.00005,
    gamma=0.99,
    ent_coef=0.05,
    clip_range=0.1,
    target_kl=0.03,
    device="cpu",
)

print()
print("Training PPO v0.4 stable CNN...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
)

final_model_path = MODEL_DIR / "ff1_nes_ppo_v0_4_final"
model.save(final_model_path)
print("Saved final model:", final_model_path.with_suffix(".zip"))

vec_env.close()

print()
print("Scoring checkpoints...")

candidate_paths = sorted(CHECKPOINT_DIR.glob("*.zip"))
candidate_paths.append(final_model_path.with_suffix(".zip"))

checkpoint_scores = []

for path in candidate_paths:
    print("Scoring:", path)
    score = score_checkpoint(path)
    checkpoint_scores.append(score)
    print(
        "  mean_reward=", score["mean_reward"],
        "mean_unique=", score["mean_unique_screens"],
        "max_repeat=", score["max_same_action_streak"],
    )

checkpoint_scores.sort(
    key=lambda item: (
        item["mean_reward"],
        item["mean_unique_screens"],
        -item["max_same_action_streak"],
    ),
    reverse=True,
)

best = checkpoint_scores[0]
best_source = Path(best["model_path"])
best_dest = MODEL_DIR / "ff1_nes_ppo_v0_4_best.zip"
shutil.copy2(best_source, best_dest)

print()
print("Best checkpoint:")
print(best_source)
print("Copied to:")
print(best_dest)
print("Best score:", best)

print()
print("Running final saved evaluation for best checkpoint...")

best_model = PPO.load(best_dest, device="cpu")

sampling_summary = run_eval(
    best_model,
    label="best_sampling",
    steps=EVAL_STEPS,
    save_frames=True,
    deterministic=False,
)

deterministic_summary = run_eval(
    best_model,
    label="best_deterministic",
    steps=EVAL_STEPS,
    save_frames=True,
    deterministic=True,
)

metrics = {
    "total_timesteps": TOTAL_TIMESTEPS,
    "policy": "CnnPolicy",
    "training_notes": {
        "learning_rate": 0.00005,
        "ent_coef": 0.05,
        "clip_range": 0.1,
        "target_kl": 0.03,
        "n_epochs": 1,
    },
    "best_model": str(best_dest),
    "checkpoint_scores": checkpoint_scores,
    "best_sampling_eval": sampling_summary,
    "best_deterministic_eval": deterministic_summary,
}

metrics_path = OUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))

print()
print("v0.4 stable training complete.")
print("Saved metrics:", metrics_path)
print("Open this folder:")
print(OUT_DIR)
print()
print("Best model saved at:")
print(best_dest)
