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
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from ff1_dawn_ai.nes_env import FF1NesEnv


TOTAL_TIMESTEPS = 10_000
EVAL_STEPS = 500
EVAL_RUNS_PER_MODEL = 5

START_MODEL = PROJECT_ROOT / "models" / "ff1_nes_ppo_v0_5_best_quality.zip"

OUT_DIR = PROJECT_ROOT / "videos" / "nes_continue_ppo_v0_6_from_quality"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "nes_continue_ppo_v0_6_from_quality"

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


def run_eval(model, label, run_index, save_frames=False):
    env = make_env(print_buttons=False, max_steps=EVAL_STEPS)
    obs, info = env.reset()

    frames = []
    total_reward = 0.0
    max_unique_screens = 0
    max_same_action_streak = 0
    max_same_screen_streak = 0
    action_counts = {name: 0 for name in env.action_names}

    for step in range(EVAL_STEPS):
        model_obs = prepare_obs_for_model(model, obs)
        action, _state = model.predict(model_obs, deterministic=False)

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
            Image.fromarray(rgb).save(OUT_DIR / f"{label}_run{run_index}_frame_{step:04d}.png")

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
        gif_path = OUT_DIR / f"{label}_run{run_index}_eval.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=80,
            loop=0,
        )
        print("Saved GIF:", gif_path)

    env.close()

    total_actions = sum(action_counts.values()) or 1
    dominant_action_fraction = max(action_counts.values()) / total_actions

    return {
        "label": label,
        "run_index": run_index,
        "total_reward": round(total_reward, 4),
        "max_unique_screens": max_unique_screens,
        "max_same_action_streak": max_same_action_streak,
        "max_same_screen_streak": max_same_screen_streak,
        "dominant_action_fraction": round(dominant_action_fraction, 4),
        "action_counts": action_counts,
        "gif": str(gif_path) if gif_path else None,
    }


def score_model(model_path):
    model = PPO.load(model_path, device="cpu")

    runs = []

    for run_index in range(EVAL_RUNS_PER_MODEL):
        result = run_eval(
            model=model,
            label=model_path.stem,
            run_index=run_index,
            save_frames=False,
        )
        runs.append(result)

    mean_reward = float(np.mean([r["total_reward"] for r in runs]))
    mean_unique = float(np.mean([r["max_unique_screens"] for r in runs]))
    worst_repeat = max(r["max_same_action_streak"] for r in runs)
    worst_screen_stuck = max(r["max_same_screen_streak"] for r in runs)
    mean_dominant_action_fraction = float(np.mean([r["dominant_action_fraction"] for r in runs]))

    quality_score = (
        mean_reward
        + (0.25 * mean_unique)
        - (6.0 * max(0, worst_repeat - 4))
        - (2.0 * max(0, worst_screen_stuck - 8))
        - (40.0 * max(0.0, mean_dominant_action_fraction - 0.45))
    )

    return {
        "model_path": str(model_path),
        "mean_reward": round(mean_reward, 4),
        "mean_unique_screens": round(mean_unique, 4),
        "worst_same_action_streak": worst_repeat,
        "worst_same_screen_streak": worst_screen_stuck,
        "mean_dominant_action_fraction": round(mean_dominant_action_fraction, 4),
        "quality_score": round(float(quality_score), 4),
        "runs": runs,
    }


print()
print("FF1 NES PPO v0.6 continue-training from v0.5 quality model")
print("----------------------------------------------------------")
print("This loads the v0.5 best-quality model, trains cautiously,")
print("then compares the starting model, checkpoints, and final model.")
print()

if not START_MODEL.exists():
    raise FileNotFoundError(f"Missing starting model: {START_MODEL}")

print("Starting model:")
print(START_MODEL)
print()

vec_env = DummyVecEnv([lambda: make_env(print_buttons=True, max_steps=500)])

checkpoint_callback = CheckpointCallback(
    save_freq=2_000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ff1_nes_ppo_v0_6_continue_checkpoint",
)

model = PPO.load(
    START_MODEL,
    env=vec_env,
    device="cpu",
)

# Cautious continue-training settings.
model.learning_rate = 0.00003
model.lr_schedule = get_schedule_fn(0.00003)
model.clip_range = get_schedule_fn(0.08)
model.ent_coef = 0.06
model.target_kl = 0.02
model.n_epochs = 1

print("Continue-training settings:")
print("learning_rate = 0.00003")
print("clip_range = 0.08")
print("ent_coef = 0.06")
print("target_kl = 0.02")
print("n_epochs = 1")
print()

print("Training PPO v0.6 continue model...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
)

final_model_path = MODEL_DIR / "ff1_nes_ppo_v0_6_continue_final"
model.save(final_model_path)
print("Saved v0.6 final model:", final_model_path.with_suffix(".zip"))

vec_env.close()

print()
print("Scoring start model, checkpoints, and final model...")

candidate_paths = [START_MODEL]
candidate_paths.extend(sorted(CHECKPOINT_DIR.glob("*.zip")))
candidate_paths.append(final_model_path.with_suffix(".zip"))

scores = []

for path in candidate_paths:
    print("Scoring:", path)
    score = score_model(path)
    scores.append(score)
    print(
        "  quality=", score["quality_score"],
        "mean_reward=", score["mean_reward"],
        "mean_unique=", score["mean_unique_screens"],
        "worst_repeat=", score["worst_same_action_streak"],
        "dominant_action=", score["mean_dominant_action_fraction"],
    )

scores.sort(
    key=lambda item: (
        item["quality_score"],
        item["mean_reward"],
        item["mean_unique_screens"],
        -item["worst_same_action_streak"],
    ),
    reverse=True,
)

best = scores[0]
best_source = Path(best["model_path"])
best_dest = MODEL_DIR / "ff1_nes_ppo_v0_6_best_continue.zip"
shutil.copy2(best_source, best_dest)

print()
print("Best v0.6 selected model:")
print(best_source)
print("Copied to:")
print(best_dest)
print()
print("Best score summary:")
print(json.dumps(
    {
        "quality_score": best["quality_score"],
        "mean_reward": best["mean_reward"],
        "mean_unique_screens": best["mean_unique_screens"],
        "worst_same_action_streak": best["worst_same_action_streak"],
        "worst_same_screen_streak": best["worst_same_screen_streak"],
        "mean_dominant_action_fraction": best["mean_dominant_action_fraction"],
    },
    indent=2,
))

print()
print("Creating proof GIF from selected best v0.6 model...")

best_loaded = PPO.load(best_dest, device="cpu")
proof = run_eval(
    model=best_loaded,
    label="v0_6_best_continue",
    run_index=0,
    save_frames=True,
)

metrics = {
    "start_model": str(START_MODEL),
    "total_timesteps": TOTAL_TIMESTEPS,
    "eval_steps": EVAL_STEPS,
    "eval_runs_per_model": EVAL_RUNS_PER_MODEL,
    "continue_training_settings": {
        "learning_rate": 0.00003,
        "clip_range": 0.08,
        "ent_coef": 0.06,
        "target_kl": 0.02,
        "n_epochs": 1,
    },
    "selection_notes": {
        "purpose": "Continue training from v0.5 but keep the starting model if training makes performance worse.",
        "quality_score_formula": "mean_reward + 0.25*mean_unique - 6*repeat_over_4 - 2*screen_stuck_over_8 - 40*dominant_action_over_45_percent",
    },
    "best_model": str(best_dest),
    "best_source": str(best_source),
    "best_score": best,
    "proof_eval": proof,
    "all_scores": scores,
}

metrics_path = OUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))

print()
print("v0.6 continue-training complete.")
print("Saved metrics:", metrics_path)
print("Open this folder:")
print(OUT_DIR)
print()
print("Best v0.6 model saved at:")
print(best_dest)
