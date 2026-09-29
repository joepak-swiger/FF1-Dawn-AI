from pathlib import Path
import json
import shutil
import sys
from datetime import datetime

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from ff1_dawn_ai.nes_env import FF1NesEnv


TOTAL_TIMESTEPS = 8_000
EVAL_STEPS = 500
EVAL_RUNS_PER_MODEL = 5

START_MODEL = PROJECT_ROOT / "models" / "ff1_nes_ppo_v0_7_best_reported.zip"

OUT_DIR = PROJECT_ROOT / "videos" / "nes_continue_ppo_v0_8_battle_aware"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "nes_continue_ppo_v0_8_battle_aware"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def log(message=""):
    print(message)


class BattleAwareFF1NesEnv(FF1NesEnv):
    """
    Experimental wrapper on top of FF1NesEnv.

    This does not truly understand Final Fantasy yet.
    It uses a visual heuristic to guess when the screen looks battle/menu-like:
    dark background, bright text/menu regions, and not much overworld green.

    Goal:
    - Keep exploration reward.
    - Add small reward for entering/exiting battle-like states.
    - Penalize long battle/menu loops.
    """

    def __init__(self, *args, **kwargs):
        self.last_battle_like = False
        self.battle_like_streak = 0
        self.battle_exit_count = 0
        super().__init__(*args, **kwargs)

    def _battle_features(self, rgb):
        arr = np.asarray(rgb).astype(np.int16)
        gray = arr[:, :, :3].mean(axis=2)

        h, w = gray.shape
        bottom = gray[int(h * 0.55):, :]
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        dark_fraction = float(np.mean(gray < 40))
        bright_fraction = float(np.mean(gray > 185))
        bottom_bright_fraction = float(np.mean(bottom > 185))

        green_pixels = (g > r + 18) & (g > b + 18) & (g > 65)
        green_fraction = float(np.mean(green_pixels))

        # Battle screens in FF1 NES are often dark with bright menu/text regions.
        # Overworld/town screens tend to have more green/tiles and less large black space.
        battle_like = (
            (dark_fraction > 0.28 and bottom_bright_fraction > 0.025 and green_fraction < 0.22)
            or
            (dark_fraction > 0.45 and bright_fraction > 0.018 and green_fraction < 0.28)
        )

        return {
            "battle_like": bool(battle_like),
            "dark_fraction": round(dark_fraction, 4),
            "bright_fraction": round(bright_fraction, 4),
            "bottom_bright_fraction": round(bottom_bright_fraction, 4),
            "green_fraction": round(green_fraction, 4),
        }

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        features = self._battle_features(self.rgb)
        self.last_battle_like = features["battle_like"]
        self.battle_like_streak = 1 if self.last_battle_like else 0
        self.battle_exit_count = 0

        info.update({
            "battle_like": self.last_battle_like,
            "battle_like_streak": self.battle_like_streak,
            "battle_exit_count": self.battle_exit_count,
            "battle_features": features,
            "battle_reward_parts": {},
        })

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        previous_battle_like = self.last_battle_like
        previous_battle_streak = self.battle_like_streak

        features = self._battle_features(self.rgb)
        battle_like = features["battle_like"]

        if battle_like:
            if previous_battle_like:
                self.battle_like_streak += 1
            else:
                self.battle_like_streak = 1
        else:
            self.battle_like_streak = 0

        battle_reward_parts = {}

        # Small bonus for detecting a transition into a battle/menu-like scene.
        if battle_like and not previous_battle_like:
            battle_reward_parts["battle_entry_bonus"] = 0.40

        # Bigger bonus for leaving a battle/menu-like scene after spending time there.
        # This is meant to reward fight resolution, running successfully, or escaping loops.
        if previous_battle_like and not battle_like and previous_battle_streak >= 8:
            self.battle_exit_count += 1
            battle_reward_parts["battle_exit_bonus"] = 4.00

        # Long battle/menu-like loops are usually bad for our current goal.
        if battle_like and self.battle_like_streak > 35:
            over = self.battle_like_streak - 35
            battle_reward_parts["long_battle_loop_penalty"] = -min(0.35, 0.015 * over)

        # Repeated A/B on the same screen is often menu thrashing.
        if battle_like and info["button"] in ("A", "B") and info["same_screen_streak"] >= 4:
            battle_reward_parts["battle_menu_loop_penalty"] = -0.18

        reward += float(sum(battle_reward_parts.values()))

        self.last_battle_like = battle_like

        info.update({
            "battle_like": battle_like,
            "battle_like_streak": self.battle_like_streak,
            "battle_exit_count": self.battle_exit_count,
            "battle_features": features,
            "battle_reward_parts": battle_reward_parts,
        })

        return obs, float(reward), terminated, truncated, info


def make_env(print_buttons=False, max_steps=500):
    return BattleAwareFF1NesEnv(
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
    max_battle_like_streak = 0
    battle_like_steps = 0
    long_battle_loop_steps = 0
    max_battle_exit_count = 0
    action_counts = {name: 0 for name in env.action_names}
    sampled_steps = []

    for step in range(EVAL_STEPS):
        model_obs = prepare_obs_for_model(model, obs)
        action, _state = model.predict(model_obs, deterministic=False)

        action_int = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = env.step(action_int)

        total_reward += reward
        max_unique_screens = max(max_unique_screens, info["unique_screens"])
        max_same_action_streak = max(max_same_action_streak, info["same_action_streak"])
        max_same_screen_streak = max(max_same_screen_streak, info["same_screen_streak"])
        max_battle_like_streak = max(max_battle_like_streak, info["battle_like_streak"])
        max_battle_exit_count = max(max_battle_exit_count, info["battle_exit_count"])
        action_counts[info["button"]] += 1

        if info["battle_like"]:
            battle_like_steps += 1

        if info["battle_like_streak"] > 35:
            long_battle_loop_steps += 1

        if step % 25 == 0:
            sampled_steps.append({
                "step": step,
                "action": info["button"],
                "reward": round(float(reward), 4),
                "total_reward": round(float(total_reward), 4),
                "unique_screens": info["unique_screens"],
                "same_action_streak": info["same_action_streak"],
                "same_screen_streak": info["same_screen_streak"],
                "battle_like": info["battle_like"],
                "battle_like_streak": info["battle_like_streak"],
                "battle_exit_count": info["battle_exit_count"],
                "battle_reward_parts": info["battle_reward_parts"],
                "battle_features": info["battle_features"],
            })

        if save_frames and step % 5 == 0:
            rgb = env.render()
            frames.append(Image.fromarray(rgb))
            Image.fromarray(rgb).save(OUT_DIR / f"{label}_run{run_index}_frame_{step:04d}.png")

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

    env.close()

    total_actions = sum(action_counts.values()) or 1
    dominant_action_fraction = max(action_counts.values()) / total_actions
    battle_like_fraction = battle_like_steps / total_actions

    return {
        "label": label,
        "run_index": run_index,
        "total_reward": round(total_reward, 4),
        "max_unique_screens": max_unique_screens,
        "max_same_action_streak": max_same_action_streak,
        "max_same_screen_streak": max_same_screen_streak,
        "max_battle_like_streak": max_battle_like_streak,
        "battle_like_steps": battle_like_steps,
        "battle_like_fraction": round(battle_like_fraction, 4),
        "long_battle_loop_steps": long_battle_loop_steps,
        "battle_exit_count": max_battle_exit_count,
        "dominant_action_fraction": round(dominant_action_fraction, 4),
        "action_counts": action_counts,
        "sampled_steps": sampled_steps,
        "gif": str(gif_path) if gif_path else None,
    }


def quality_score_from_runs(runs):
    mean_reward = float(np.mean([r["total_reward"] for r in runs]))
    mean_unique = float(np.mean([r["max_unique_screens"] for r in runs]))
    worst_repeat = max(r["max_same_action_streak"] for r in runs)
    worst_screen_stuck = max(r["max_same_screen_streak"] for r in runs)
    worst_battle_streak = max(r["max_battle_like_streak"] for r in runs)
    mean_battle_like_fraction = float(np.mean([r["battle_like_fraction"] for r in runs]))
    mean_long_battle_loop_steps = float(np.mean([r["long_battle_loop_steps"] for r in runs]))
    mean_battle_exits = float(np.mean([r["battle_exit_count"] for r in runs]))
    mean_dominant_action_fraction = float(np.mean([r["dominant_action_fraction"] for r in runs]))

    quality_score = (
        mean_reward
        + (0.25 * mean_unique)
        + (12.0 * mean_battle_exits)
        - (6.0 * max(0, worst_repeat - 4))
        - (2.0 * max(0, worst_screen_stuck - 8))
        - (1.5 * max(0, worst_battle_streak - 45))
        - (1.0 * mean_long_battle_loop_steps)
        - (40.0 * max(0.0, mean_dominant_action_fraction - 0.45))
        - (25.0 * max(0.0, mean_battle_like_fraction - 0.65))
    )

    return {
        "mean_reward": round(mean_reward, 4),
        "mean_unique_screens": round(mean_unique, 4),
        "worst_same_action_streak": worst_repeat,
        "worst_same_screen_streak": worst_screen_stuck,
        "worst_battle_like_streak": worst_battle_streak,
        "mean_battle_like_fraction": round(mean_battle_like_fraction, 4),
        "mean_long_battle_loop_steps": round(mean_long_battle_loop_steps, 4),
        "mean_battle_exits": round(mean_battle_exits, 4),
        "mean_dominant_action_fraction": round(mean_dominant_action_fraction, 4),
        "quality_score": round(float(quality_score), 4),
    }


def score_model(model_path):
    model = PPO.load(model_path, device="cpu")
    runs = []

    for run_index in range(EVAL_RUNS_PER_MODEL):
        runs.append(run_eval(
            model=model,
            label=model_path.stem,
            run_index=run_index,
            save_frames=False,
        ))

    score = quality_score_from_runs(runs)
    score["model_path"] = str(model_path)
    score["runs"] = runs
    return score


def write_reports(metrics):
    best = metrics["best_score"]
    proof = metrics["proof_eval"]

    summary_lines = [
        "FF1 NES PPO v0.8 Battle-Aware Continue Training",
        "================================================",
        "",
        f"Created: {metrics['created_at']}",
        f"Start model: {metrics['start_model']}",
        f"Best selected model: {metrics['best_model']}",
        f"Best source: {metrics['best_source']}",
        "",
        "Important note",
        "--------------",
        "Battle awareness is heuristic in this version.",
        "It guesses battle/menu-like screens from image features.",
        "This is useful for experiments, but it is not true game-state memory yet.",
        "",
        "Training settings",
        "-----------------",
        f"total_timesteps: {metrics['total_timesteps']}",
        f"learning_rate: {metrics['continue_training_settings']['learning_rate']}",
        f"clip_range: {metrics['continue_training_settings']['clip_range']}",
        f"ent_coef: {metrics['continue_training_settings']['ent_coef']}",
        f"target_kl: {metrics['continue_training_settings']['target_kl']}",
        f"n_epochs: {metrics['continue_training_settings']['n_epochs']}",
        "",
        "Best score summary",
        "------------------",
        f"quality_score: {best['quality_score']}",
        f"mean_reward: {best['mean_reward']}",
        f"mean_unique_screens: {best['mean_unique_screens']}",
        f"worst_same_action_streak: {best['worst_same_action_streak']}",
        f"worst_same_screen_streak: {best['worst_same_screen_streak']}",
        f"worst_battle_like_streak: {best['worst_battle_like_streak']}",
        f"mean_battle_like_fraction: {best['mean_battle_like_fraction']}",
        f"mean_long_battle_loop_steps: {best['mean_long_battle_loop_steps']}",
        f"mean_battle_exits: {best['mean_battle_exits']}",
        f"mean_dominant_action_fraction: {best['mean_dominant_action_fraction']}",
        "",
        "Proof eval summary",
        "------------------",
        f"total_reward: {proof['total_reward']}",
        f"max_unique_screens: {proof['max_unique_screens']}",
        f"max_same_action_streak: {proof['max_same_action_streak']}",
        f"max_same_screen_streak: {proof['max_same_screen_streak']}",
        f"max_battle_like_streak: {proof['max_battle_like_streak']}",
        f"battle_like_fraction: {proof['battle_like_fraction']}",
        f"long_battle_loop_steps: {proof['long_battle_loop_steps']}",
        f"battle_exit_count: {proof['battle_exit_count']}",
        f"dominant_action_fraction: {proof['dominant_action_fraction']}",
        f"action_counts: {proof['action_counts']}",
        f"gif: {proof['gif']}",
        "",
        "Checkpoint ranking",
        "------------------",
    ]

    for index, score in enumerate(metrics["all_scores"], start=1):
        summary_lines.append(
            f"{index}. quality={score['quality_score']} "
            f"reward={score['mean_reward']} "
            f"unique={score['mean_unique_screens']} "
            f"repeat={score['worst_same_action_streak']} "
            f"battle_streak={score['worst_battle_like_streak']} "
            f"battle_fraction={score['mean_battle_like_fraction']} "
            f"battle_exits={score['mean_battle_exits']} "
            f"dominant={score['mean_dominant_action_fraction']} "
            f"path={score['model_path']}"
        )

    summary_lines.extend([
        "",
        "Proof eval sampled steps",
        "------------------------",
    ])

    for item in proof["sampled_steps"]:
        summary_lines.append(
            f"step={item['step']:04d} "
            f"action={item['action']} "
            f"reward={item['reward']} "
            f"total={item['total_reward']} "
            f"unique={item['unique_screens']} "
            f"same_action={item['same_action_streak']} "
            f"same_screen={item['same_screen_streak']} "
            f"battle_like={item['battle_like']} "
            f"battle_streak={item['battle_like_streak']} "
            f"battle_exits={item['battle_exit_count']} "
            f"battle_reward={item['battle_reward_parts']} "
            f"features={item['battle_features']}"
        )

    run_report_path = OUT_DIR / "run_report.txt"
    run_report_path.write_text("\n".join(summary_lines) + "\n")

    best_model_summary_path = OUT_DIR / "best_model_summary.txt"
    best_model_summary_path.write_text(
        "\n".join([
            "Best model summary",
            "==================",
            "",
            f"Best model: {metrics['best_model']}",
            f"Best source: {metrics['best_source']}",
            f"Quality score: {best['quality_score']}",
            f"Mean reward: {best['mean_reward']}",
            f"Mean unique screens: {best['mean_unique_screens']}",
            f"Worst same action streak: {best['worst_same_action_streak']}",
            f"Worst same screen streak: {best['worst_same_screen_streak']}",
            f"Worst battle-like streak: {best['worst_battle_like_streak']}",
            f"Mean battle-like fraction: {best['mean_battle_like_fraction']}",
            f"Mean long battle loop steps: {best['mean_long_battle_loop_steps']}",
            f"Mean battle exits: {best['mean_battle_exits']}",
            f"Mean dominant action fraction: {best['mean_dominant_action_fraction']}",
            f"Proof GIF: {proof['gif']}",
        ]) + "\n"
    )

    return run_report_path, best_model_summary_path


log()
log("FF1 NES PPO v0.8 battle-aware continue-training")
log("------------------------------------------------")
log("This run writes clean reports and uses heuristic battle/menu detection.")
log()

if not START_MODEL.exists():
    raise FileNotFoundError(f"Missing starting model: {START_MODEL}")

log(f"Starting model: {START_MODEL}")
log(f"Output folder: {OUT_DIR}")
log()

vec_env = DummyVecEnv([lambda: make_env(print_buttons=False, max_steps=500)])

checkpoint_callback = CheckpointCallback(
    save_freq=2_000,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ff1_nes_ppo_v0_8_battle_aware_checkpoint",
)

model = PPO.load(
    START_MODEL,
    env=vec_env,
    device="cpu",
)

model.learning_rate = 0.000015
model.lr_schedule = get_schedule_fn(0.000015)
model.clip_range = get_schedule_fn(0.05)
model.ent_coef = 0.08
model.target_kl = 0.012
model.n_epochs = 1
model.verbose = 0

log("Training quietly. This may take a minute or two...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
)

final_model_path = MODEL_DIR / "ff1_nes_ppo_v0_8_battle_aware_final"
model.save(final_model_path)
vec_env.close()

log("Training finished.")
log(f"Saved final model: {final_model_path.with_suffix('.zip')}")
log()
log("Scoring start model, checkpoints, and final model...")

candidate_paths = [START_MODEL]
candidate_paths.extend(sorted(CHECKPOINT_DIR.glob("*.zip")))
candidate_paths.append(final_model_path.with_suffix(".zip"))

scores = []

for path in candidate_paths:
    score = score_model(path)
    scores.append(score)
    log(
        f"score={score['quality_score']} "
        f"reward={score['mean_reward']} "
        f"unique={score['mean_unique_screens']} "
        f"battle_streak={score['worst_battle_like_streak']} "
        f"battle_exits={score['mean_battle_exits']} "
        f"model={path.name}"
    )

scores.sort(
    key=lambda item: (
        item["quality_score"],
        item["mean_reward"],
        item["mean_unique_screens"],
        item["mean_battle_exits"],
        -item["worst_battle_like_streak"],
    ),
    reverse=True,
)

best = scores[0]
best_source = Path(best["model_path"])
best_dest = MODEL_DIR / "ff1_nes_ppo_v0_8_best_battle_aware.zip"
shutil.copy2(best_source, best_dest)

log()
log("Creating proof GIF from selected best v0.8 model...")

best_loaded = PPO.load(best_dest, device="cpu")
proof = run_eval(
    model=best_loaded,
    label="v0_8_best_battle_aware",
    run_index=0,
    save_frames=True,
)

metrics = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "start_model": str(START_MODEL),
    "total_timesteps": TOTAL_TIMESTEPS,
    "eval_steps": EVAL_STEPS,
    "eval_runs_per_model": EVAL_RUNS_PER_MODEL,
    "continue_training_settings": {
        "learning_rate": 0.000015,
        "clip_range": 0.05,
        "ent_coef": 0.08,
        "target_kl": 0.012,
        "n_epochs": 1,
    },
    "selection_notes": {
        "purpose": "Continue training from v0.7 while adding heuristic battle/menu awareness.",
        "warning": "Battle detection is image-based and imperfect. Later versions should use RAM/game-state detection.",
        "quality_score_formula": "mean_reward + 0.25*mean_unique + 12*battle_exits - repeat/stuck/battle-loop/dominant-action penalties",
    },
    "best_model": str(best_dest),
    "best_source": str(best_source),
    "best_score": best,
    "proof_eval": proof,
    "all_scores": scores,
}

metrics_path = OUT_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))

run_report_path, best_model_summary_path = write_reports(metrics)

log()
log("v0.8 battle-aware continue-training complete.")
log(f"Best model: {best_dest}")
log(f"Metrics: {metrics_path}")
log(f"Run report: {run_report_path}")
log(f"Best model summary: {best_model_summary_path}")
log(f"Output folder: {OUT_DIR}")
