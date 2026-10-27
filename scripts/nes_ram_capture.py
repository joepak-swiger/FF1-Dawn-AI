"""Headless multi-session RAM capture using an existing PPO policy.

This never trains or modifies the model. Battle/field labels come solely from
the frame-pixel heuristic, never from RAM candidates; review saved frames when
the automated visual signal is uncertain.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from ff1_dawn_ai.nes_env import FF1NesEnv
from ff1_dawn_ai.ram_analysis import pixel_scene_features

DEFAULT_MODEL = PROJECT_ROOT / "models" / "ff1_nes_ppo_v0_7_best_reported.zip"
OUTPUT_ROOT = PROJECT_ROOT / "videos" / "nes_ram_auto_capture"
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]


def read_ram(env):
    candidates = [env, getattr(env, "retro_env", None), getattr(getattr(env, "retro_env", None), "unwrapped", None)]
    for obj in candidates:
        if obj is None:
            continue
        getter = getattr(obj, "get_ram", None)
        if callable(getter):
            ram = np.asarray(getter(), dtype=np.uint8).reshape(-1)
            if ram.size:
                return ram
        raw = getattr(obj, "ram", None)
        if raw is not None:
            ram = np.asarray(raw, dtype=np.uint8).reshape(-1)
            if ram.size:
                return ram
    raise RuntimeError("Stable-Retro did not expose RAM through the known access paths")


def screen_hash(rgb):
    return hashlib.sha1(np.asarray(rgb)[::8, ::8, :3].tobytes()).hexdigest()[:12]


def policy_observation(model, obs):
    obs = np.asarray(obs)
    shape = model.policy.observation_space.shape
    if obs.shape == shape:
        return obs
    if len(obs.shape) == 3 and len(shape) == 3 and obs.shape == (shape[1], shape[2], shape[0]):
        return np.transpose(obs, (2, 0, 1))
    return obs


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["step"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture_episode(env, model, session_dir, steps, frame_every, deterministic, seed):
    obs, _ = env.reset(seed=seed)
    ram_rows, visual_labels, event_rows = [], [], []
    prev_action = "RESET"
    total_reward = 0.0
    previous_visual = None
    visual_transitions = []
    failure = None
    try:
        for step in range(steps + 1):
            rgb = env.render()
            ram = read_ram(env)
            features = pixel_scene_features(rgb)
            visual_labels.append(features["battle_like"])
            ram_rows.append(ram.copy())
            if previous_visual is not None and features["battle_like"] != previous_visual:
                visual_transitions.append({
                    "step": step,
                    "from": "battle_like" if previous_visual else "field_like",
                    "to": "battle_like" if features["battle_like"] else "field_like",
                })
            previous_visual = features["battle_like"]
            event_rows.append({
                "step": step,
                "prev_action": prev_action,
                "total_reward": round(total_reward, 4),
                "screen_hash": screen_hash(rgb),
                "battle_like": features["battle_like"],
                "label_source": "visual_heuristic_v0_9_pixels_only",
                **{key: value for key, value in features.items() if key != "battle_like"},
            })
            if step and step % 100 == 0:
                print(f"{session_dir.name}: captured {step}/{steps} decisions", flush=True)
            if step % frame_every == 0:
                Image.fromarray(rgb).save(session_dir / "frames" / f"frame_{step:05d}_{event_rows[-1]['screen_hash']}.png")
            if step == steps:
                break
            action, _ = model.predict(policy_observation(model, obs), deterministic=deterministic)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            prev_action = info.get("button", ACTION_NAMES[action])
            if terminated or truncated:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        np.savez_compressed(
            session_dir / "ram_samples.npz",
            rams=np.asarray(ram_rows, dtype=np.uint8),
            battle_like=np.asarray(visual_labels, dtype=bool),
        )
        write_csv(session_dir / "probe_events.csv", event_rows)
        metrics = {
            "session_id": session_dir.name,
            "status": "complete" if failure is None else "partial",
            "steps_recorded": len(event_rows),
            "requested_steps": steps + 1,
            "ram_size": int(ram_rows[0].size) if ram_rows else 0,
            "visual_battle_like_samples": int(sum(visual_labels)),
            "visual_field_like_samples": int(len(visual_labels) - sum(visual_labels)),
            "total_reward": round(total_reward, 4),
            "visual_transitions": visual_transitions,
            "label_source": "independent frame-pixel heuristic; not a verified game-state label",
            "model_is_unmodified": True,
        }
        if failure:
            metrics["failure"] = failure
        (session_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if failure:
        raise RuntimeError(f"{session_dir.name} failed after {len(event_rows)} samples: {failure}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=3, help="Independent resets from the configured Cornelia state")
    parser.add_argument("--steps", type=int, default=1800, help="Maximum environment decisions per episode")
    parser.add_argument("--frame-every", type=int, default=40)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--deterministic", action="store_true", help="Use the policy's deterministic action")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true", help="Validate runtime assets without loading the model or starting the emulator")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.frame_every < 1:
        parser.error("--episodes, --steps, and --frame-every must be positive")
    if not args.model.is_absolute():
        args.model = PROJECT_ROOT / args.model
    if not args.model.exists():
        parser.error(f"Model not found: {args.model}")
    rom = PROJECT_ROOT / "custom_integrations" / "FinalFantasy-Nes" / "rom.nes"
    state = PROJECT_ROOT / "states" / "ff1_nes_cornelia_start.state"
    if not rom.is_file():
        parser.error(f"NES ROM is missing from local integration: {rom}")
    if not state.is_file():
        parser.error(f"Cornelia start state is missing: {state}")
    if args.dry_run:
        print("Dry run passed: capture dependencies imported, v0.7 model/ROM/state exist.")
        print(f"Model: {args.model}")
        print(f"Capture plan: {args.episodes} independent resets x {args.steps} decisions")
        print("No emulator was started and no capture output was created.")
        return 0
    run_id = args.run_id or (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        parser.error("--run-id must be 1-80 letters, digits, dots, underscores, or hyphens")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    sessions = []
    failure = None
    try:
        model = PPO.load(args.model, device="cpu")
        set_random_seed(args.seed)
        env = FF1NesEnv(project_root=PROJECT_ROOT, max_steps=args.steps + 1, action_freq=16, print_buttons=False)
        for index in range(args.episodes):
            session_dir = run_dir / f"session_{index + 1:02d}"
            (session_dir / "frames").mkdir(parents=True)
            result = capture_episode(
                env, model, session_dir, args.steps, args.frame_every,
                args.deterministic, args.seed + index,
            )
            sessions.append(result)
            print(f"{session_dir.name}: {result['steps_recorded']} steps, visual battles={result['visual_battle_like_samples']}")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"Capture failed: {failure}", file=sys.stderr)
    finally:
        if "env" in locals():
            env.close()
    summary = {
        "run_id": run_id,
        "status": "complete" if failure is None and len(sessions) == args.episodes else "failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "model_is_unmodified": True,
        "episodes": sessions,
        "requested_episodes": args.episodes,
        "requested_steps_per_episode": args.steps + 1,
        "failure": failure,
        "interpretation": "Pixel-only scene labels are proxies; inspect saved proof frames before treating transitions as ground truth.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["Automated NES RAM capture", "=========================", f"Run ID: {run_id}", f"Status: {summary['status']}",
              f"Model: {args.model.name}", f"Independent resets: {len(sessions)}", ""]
    for item in sessions:
        report.append(
            f"{item['session_id']}: steps={item['steps_recorded']} battle_like={item['visual_battle_like_samples']} "
            f"field_like={item['visual_field_like_samples']} visual_transitions={len(item['visual_transitions'])}"
        )
    report.extend(["", "Labels come from the screen-pixel heuristic only and require independent visual review."])
    (run_dir / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Automated capture written to {run_dir}")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
