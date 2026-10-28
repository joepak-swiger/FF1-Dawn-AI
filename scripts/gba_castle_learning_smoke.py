"""Bounded PPO feasibility smoke for the optional Castle-entry goal.

This is a small sparse-reward feasibility trial, not evidence of learned
navigation. It compares scripted UP, seeded random, idle, and a short PPO run
from the same fixed checkpoint. Model weights and raw traces stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402
from ff1_dawn_ai.gba_goal_reward import FirstCastleEntryReward  # noqa: E402
from ff1_dawn_ai.gba_landmark import SceneAnchorClassifier  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402

EVIDENCE = ROOT / "docs/experiments/gba_castle_goal/20261028_052724_624774Z"
REFERENCES = {
    "castle_cornelia_1f": ROOT / "docs/experiments/gba_castle_goal/reference_cornelia_1f.png",
    "cornelia_exterior_field": EVIDENCE / "overworld_after_castle_exit.png",
    "cornelia_town": EVIDENCE / "town_ram_collision.png",
    "menu": EVIDENCE / "party_menu_negative_control.png",
}
FIELD_VARIANTS = (ROOT / "docs/experiments/gba_scene_classifier/cornelia_exterior_checkpoint_anchor.png",)


def make_classifier() -> SceneAnchorClassifier:
    return SceneAnchorClassifier(REFERENCES, field_reference_variants=FIELD_VARIANTS)


def make_env(checkpoint: Path, *, action_frames: int, episode_steps: int,
             classifier: SceneAnchorClassifier | None = None, enabled: bool = True):
    base = FF1GbaEnv(checkpoint, project_root=ROOT, max_steps=episode_steps,
                     action_frames=action_frames, render_mode="rgb_array", restore_count=2)
    return FirstCastleEntryReward(base, classifier or make_classifier(), enabled=enabled)


def run_policy(checkpoint: Path, name: str, *, episodes: int, action_frames: int,
               episode_steps: int, seeds: tuple[int, ...], model=None,
               save_dir: Path | None = None) -> list[dict]:
    env = make_env(checkpoint, action_frames=action_frames, episode_steps=episode_steps,
                   enabled=True)
    rows = []
    try:
        for index in range(episodes):
            seed = seeds[index % len(seeds)]
            rng = np.random.default_rng(seed)
            obs, reset_info = env.reset(seed=seed)
            actions: dict[str, int] = {}
            goal = False
            goal_step = None
            total_reward = 0.0
            frames = 0
            saved = []
            for step in range(1, episode_steps + 1):
                if name == "SCRIPTED_UP":
                    action = GBA_ACTIONS.index("UP")
                elif name == "IDLE":
                    action = GBA_ACTIONS.index("NOOP")
                elif name == "SEEDED_RANDOM":
                    action = int(rng.integers(0, 5))
                elif name == "PPO_TRAINED":
                    action, _state = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    raise ValueError(name)
                obs, reward, terminated, truncated, info = env.step(action)
                frames += info.get("emulator_frames_executed", action_frames)
                total_reward += float(reward)
                action_name = info.get("action", GBA_ACTIONS[action])
                actions[action_name] = actions.get(action_name, 0) + 1
                if len(saved) < 7 and step in {1, 2, 4, 8, 12, 16, 24}:
                    saved.append((step, env.render().copy()))
                if info.get("first_castle_entry_event"):
                    goal = True
                    goal_step = step
                    if save_dir is not None:
                        Image.fromarray(env.render()).save(save_dir / f"{name.lower()}_{index+1:02d}_goal.png")
                if terminated or truncated:
                    break
            if save_dir is not None:
                for frame_step, frame in saved:
                    Image.fromarray(frame).save(save_dir / f"{name.lower()}_{index+1:02d}_step_{frame_step:03d}.png")
            rows.append({"policy": name, "episode": index + 1, "seed": seed,
                         "goal_reached": goal, "first_goal_step": goal_step,
                         "decision_steps": step, "emulator_frames": frames,
                         "total_reward": round(total_reward, 4), "action_counts": actions,
                         "checkpoint_loaded": bool(reset_info.get("checkpoint_loaded")),
                         "ended_on_goal": bool(goal and truncated)})
    finally:
        env.close()
    return rows


def _make_contact_sheet(out: Path, episode_rows: list[dict]) -> None:
    # Keep the published proof concise: three decision points from the first
    # held-out episode for each policy, plus the goal frame for the scripted
    # reference if present. Avoid turning a short experiment into a wall of
    # highly correlated screenshots.
    candidates = []
    for policy in ("SCRIPTED_UP", "SEEDED_RANDOM", "IDLE", "PPO_TRAINED"):
        for step in (1, 8, 16):
            path = out / f"{policy.lower()}_01_step_{step:03d}.png"
            if path.is_file():
                candidates.append(path)
    tile_w, tile_h, cols = 260, 190, 3
    sheet = Image.new("RGB", (cols * tile_w, ((len(candidates) + cols - 1) // cols) * tile_h), "#10182b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, path in enumerate(candidates):
        x, y = index % cols * tile_w, index // cols * tile_h
        with Image.open(path) as image:
            thumb = image.convert("RGB").resize((240, 160), Image.Resampling.NEAREST)
        sheet.paste(thumb, (x + 8, y + 22))
        draw.text((x + 8, y + 4), path.stem[:36], fill="white", font=font)
    sheet.save(out / "policy_contact_sheet.png", optimize=True)


def run(checkpoint: Path, *, total_timesteps: int = 2048,
        evaluation_episodes: int = 3) -> Path:
    if total_timesteps < 512 or total_timesteps > 4096:
        raise ValueError("bounded PPO smoke budget must be between 512 and 4096 timesteps")
    if evaluation_episodes < 3:
        raise ValueError("use at least 3 evaluation episodes per policy")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for path in (*REFERENCES.values(), *FIELD_VARIANTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    parent = ROOT / "videos" / "gba_castle_learning"
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out.mkdir()
    started = time.perf_counter()

    # Same discrete controls and frame repeat for all baselines/evaluation.
    action_frames, episode_steps = 8, 32
    heldout_seeds = (310, 311, 312)
    all_rows = []
    for policy in ("SCRIPTED_UP", "SEEDED_RANDOM", "IDLE"):
        all_rows.extend(run_policy(checkpoint, policy, episodes=evaluation_episodes,
                                   action_frames=action_frames, episode_steps=episode_steps,
                                   seeds=heldout_seeds, save_dir=out))

    # Small CPU-only PPO from scratch. Sparse reward is intentionally the only
    # learning signal. A positive result remains a fixed-start feasibility
    # observation; no hardcoded action sequence is included in the policy.
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is required for the bounded PPO smoke") from exc

    train_env = make_env(checkpoint, action_frames=action_frames,
                         episode_steps=episode_steps, enabled=True)

    class GoalCounter(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.goal_events = 0

        def _on_step(self) -> bool:
            self.goal_events += sum(bool(info.get("first_castle_entry_event"))
                                    for info in self.locals.get("infos", []))
            return True

    callback = GoalCounter()
    model = PPO("CnnPolicy", train_env, seed=1729, device="cpu", verbose=0,
                n_steps=128, batch_size=64, n_epochs=2, learning_rate=2.5e-4,
                gamma=0.99, ent_coef=0.01)
    training_started = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    training_seconds = time.perf_counter() - training_started
    model_path = out / "ppo_bounded_smoke.zip"
    model.save(model_path)
    train_env.close()
    all_rows.extend(run_policy(checkpoint, "PPO_TRAINED", episodes=evaluation_episodes,
                               action_frames=action_frames, episode_steps=episode_steps,
                               seeds=heldout_seeds, model=model, save_dir=out))
    _make_contact_sheet(out, all_rows)

    summary = {}
    for policy in ("SCRIPTED_UP", "SEEDED_RANDOM", "IDLE", "PPO_TRAINED"):
        rows = [row for row in all_rows if row["policy"] == policy]
        summary[policy] = {"episodes": len(rows), "successes": sum(row["goal_reached"] for row in rows),
                           "success_rate": round(sum(row["goal_reached"] for row in rows) / len(rows), 4),
                           "steps_to_goal": [row["first_goal_step"] for row in rows if row["goal_reached"]],
                           "mean_decision_steps": round(float(np.mean([r["decision_steps"] for r in rows])), 3),
                           "action_counts": {action: sum(r["action_counts"].get(action, 0) for r in rows)
                                             for action in GBA_ACTIONS}}
    metrics = {
        "run_id": out.name, "checkpoint_run_id": checkpoint.parent.name,
        "checkpoint_sha256_unchanged": hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash,
        "reward": {"type": "one-time first confirmed exterior-to-Castle transition",
                   "value": 1.0, "disabled_in_base_env": True, "RAM_used": False,
                   "single_frame_match_rewarded": False, "goal_ends_episode": True},
        "scene_gate": {"consecutive_frames": 2, "entry_requires_confirmed_exterior": True,
                       "unknown_frames_rewarded": False},
        "training": {"algorithm": "Stable-Baselines3 PPO", "policy": "CnnPolicy",
                      "timesteps_requested": total_timesteps, "timesteps_actual": int(model.num_timesteps),
                      "timesteps_per_episode_limit": episode_steps,
                      "emulator_frames_per_action": action_frames,
                      "episodes_with_goal_events": callback.goal_events,
                      "elapsed_seconds": round(training_seconds, 3),
                      "device": "cpu", "from_scratch": True,
                      "model_file_local_only": model_path.name},
        "evaluation": summary, "episodes": all_rows,
        "claim": "bounded feasibility smoke only; a success would not establish general navigation or story learning",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = ["# Bounded GBA Castle-entry learning feasibility smoke", "",
              f"Run `{out.name}` used the verified checkpoint `{checkpoint.parent.name}`.", "",
              f"PPO trained from scratch for {model.num_timesteps} actual timesteps in {training_seconds:.1f} seconds on CPU.",
              f"The training collection observed {callback.goal_events} confirmed goal events.", "",
              "## Same-start held-out comparison", ""]
    for policy, result in summary.items():
        report.append(f"- {policy}: {result['successes']}/{result['episodes']} confirmed first-entry goals; steps={result['steps_to_goal']}.")
    report += ["", "The reward is optional and isolated in a wrapper; FF1GbaEnv remains neutral. It uses only a two-frame confirmed visual transition from a confirmed Cornelia exterior anchor to the in-game Castle floor title. No RAM addresses or pixel-change rewards are used.",
               "", "This sparse-reward run is a feasibility check, not evidence of learned navigation. Evaluations share one fixed checkpoint/start, three short episodes per policy, and a 32-decision horizon. The scripted north policy is a reference baseline, not an AI result. Raw traces and PPO weights remain local."]
    (out / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint or discover_checkpoint()
    if args.dry_run:
        print(json.dumps({"runtime_not_started": True, "checkpoint_bytes": checkpoint.stat().st_size,
                          "timesteps": args.timesteps, "episodes_per_policy": args.episodes,
                          "allowed_budget": "512..4096"}, indent=2))
        return 0
    print(run(checkpoint, total_timesteps=args.timesteps,
              evaluation_episodes=args.episodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
