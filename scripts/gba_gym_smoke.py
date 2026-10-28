"""Run a short deterministic, reward-neutral GBA Gymnasium gameplay smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402


def discover_checkpoint() -> Path:
    candidates = []
    for metrics_path in (ROOT / "videos" / "gba_feasibility").glob("*/metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            runtime = metrics.get("runtime", {})
            machine = runtime.get("party_state_machine", {})
            names = machine.get("entered_names", [])
            roster = [(item.get("canonical_name"), item.get("display_name")) for item in names]
            classes = machine.get("classes")
            state = ROOT / "states" / "gba_custom_party" / metrics_path.parent.name / "stable_retro_core.state"
            if (runtime.get("gameplay_readiness") == "verified"
                    and roster == [("Sif", "Sif"), ("Myth", "Myth"), ("Milo", "Milo"), ("Seed", "Seed")]
                    and classes == ["Warrior", "Thief", "Monk", "Red Mage"]
                    and state.is_file()):
                candidates.append((state.stat().st_mtime_ns, state))
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        raise FileNotFoundError(
            "No verified Seed/Red Mage custom-party checkpoint found. "
            "Complete run_gba_feasibility.cmd first or pass --checkpoint explicitly."
        )
    return max(candidates)[1]


def new_output(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    path = root / stamp
    path.mkdir()
    return path


def seeded_direction_actions(steps: int, seed: int) -> list[str]:
    """Small reproducible random policy over cardinal movement controls."""
    rng = np.random.default_rng(seed)
    actions = ("UP", "DOWN", "LEFT", "RIGHT")
    return [actions[int(index)] for index in rng.integers(0, len(actions), size=steps)]


def longest_repeat_streak(hashes: list[str]) -> int:
    """Count consecutive identical observations as a diagnostic, not a stall label."""
    longest = current = 0
    previous = None
    for digest in hashes:
        current = current + 1 if digest == previous else 1
        longest = max(longest, current)
        previous = digest
    return longest


def make_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    width = 3 * 260 + 20
    height = ((len(frames) + 2) // 3) * 190 + 10
    sheet = Image.new("RGB", (width, height), "#10182b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, frame in enumerate(frames):
        x = 10 + (i % 3) * 260
        y = 8 + (i // 3) * 190
        image = Image.fromarray(frame).resize((240, 160), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y + 18))
        draw.text((x, y), f"checkpoint/action {i}", fill="white", font=font)
    sheet.save(path, optimize=True)


def package_public_evidence(run_dir: Path, destination: Path | None = None) -> Path:
    """Publish only validated, sanitized smoke metrics and one small proof image."""
    run_dir = run_dir.resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if not (
        metrics.get("checkpoint_loaded")
        and metrics.get("reset_reproducible_observation")
        and metrics.get("neutral_rewards_only")
        and metrics.get("checkpoint_source_unchanged")
        and metrics.get("completed_steps", 0) == metrics.get("requested_steps", -1)
        and metrics.get("completed_steps", 0) > 0
        and metrics.get("episodes_reproducible")
        and metrics.get("episodes_completed") == metrics.get("episodes_requested")
        and (metrics.get("terminated") or metrics.get("truncated"))
    ):
        raise ValueError("Refusing to publish an incomplete or nondeterministic GBA Gymnasium smoke")
    proof = run_dir / "contact_sheet.png"
    if not proof.is_file():
        raise FileNotFoundError(proof)
    target = destination or ROOT / "docs" / "experiments" / "gba_gymnasium" / run_dir.name
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=False)
    public_metrics = {
        key: metrics[key] for key in (
            "run_id", "checkpoint_run_id", "checkpoint_loaded", "reset_reproducible_observation",
            "observation_shape", "observation_dtype", "action_names",
            "requested_steps", "steps_per_episode", "completed_steps", "action_frames",
            "episodes_requested", "episodes_completed", "episodes_reproducible",
            "action_counts", "longest_identical_observation_streak",
            "neutral_rewards_only", "terminated", "truncated",
            "checkpoint_source_unchanged", "limitations",
        )
    }
    (target / "metrics.json").write_text(json.dumps(public_metrics, indent=2) + "\n", encoding="utf-8")
    image = Image.open(proof).convert("RGB")
    if image.width > 600:
        image.thumbnail((600, 1200), Image.Resampling.LANCZOS)
    image.save(target / "proof.png", optimize=True)
    report = [
        "# GBA Gymnasium gameplay smoke", "",
        f"Run {metrics['run_id']} restored the Seed/Red Mage native checkpoint and completed {metrics['completed_steps']} discrete actions over {metrics['episodes_completed']} reproducible episodes.",
        "",
        f"- Reset observations reproducible: {metrics['reset_reproducible_observation']}",
        f"- Observation shape and dtype: {metrics['observation_shape']} / {metrics['observation_dtype']}",
        f"- Neutral rewards on every action: {metrics['neutral_rewards_only']}",
        f"- Episode result: terminated={metrics['terminated']}, truncated={metrics['truncated']}",
        "- Source native checkpoint remained unchanged.",
        f"- Action counts: {metrics['action_counts']}",
        f"- Longest consecutive identical observation streak: {metrics['longest_identical_observation_streak']} (diagnostic only; not a semantic stuck detector).",
        "",
        "The contact sheet shows the Cornelia scene and response frames. This smoke validates the Gymnasium/core interface and input execution; it does not infer coordinates, battle outcomes, or story progress. No PPO training ran.",
    ]
    (target / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return target


def run(checkpoint: Path | None, steps: int, action_frames: int, max_steps: int | None = None, episodes: int = 2) -> Path:
    checkpoint = (checkpoint or discover_checkpoint()).resolve()
    output = new_output(ROOT / "videos" / "gba_gym_smoke")
    env = FF1GbaEnv(
        checkpoint,
        project_root=ROOT,
        max_steps=max_steps or steps,
        action_frames=action_frames,
        render_mode="rgb_array",
    )
    rows: list[dict] = []
    proof_frames: list[np.ndarray] = []
    try:
        action_sequence = seeded_direction_actions(steps, seed=1729)
        episode_hashes: list[list[str]] = []
        action_counts = {name: 0 for name in ("UP", "DOWN", "LEFT", "RIGHT")}
        terminated = truncated = False
        reset_info = {}
        for episode in range(episodes):
            obs, reset_info = env.reset(seed=1729)
            hashes = [hashlib.sha256(obs.tobytes()).hexdigest()]
            if episode == 0:
                proof_frames.append(env.render())
            for step, action_name in enumerate(action_sequence):
                action = GBA_ACTIONS.index(action_name)
                obs, reward, terminated, truncated, info = env.step(action)
                digest = hashlib.sha256(obs.tobytes()).hexdigest()
                hashes.append(digest)
                frame = env.render()
                image_name = f"episode_{episode + 1}_frame_{step + 1:03d}_{action_name.lower()}.png"
                if episode == 0 and (step < 12 or step == steps - 1):
                    Image.fromarray(frame).save(output / image_name)
                    proof_frames.append(frame)
                action_counts[action_name] += 1
                rows.append({
                    "episode": episode + 1, "step": step + 1,
                    "action_index": action, "action": action_name,
                    "reward": reward, "terminated": terminated, "truncated": truncated,
                    "observation_sha256": digest,
                    "frame": image_name if episode == 0 and (step < 12 or step == steps - 1) else "",
                })
                if terminated or truncated:
                    break
            episode_hashes.append(hashes)
            # Time-limit truncation ends this episode but should not prevent
            # the requested clean checkpoint reset and reproducibility trial.
            if terminated:
                break
        with (output / "action_trace.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["episode", "step", "action"])
            writer.writeheader()
            writer.writerows(rows)
        make_contact_sheet(proof_frames, output / "contact_sheet.png")
        images = [Image.fromarray(frame).resize((240, 160), Image.Resampling.NEAREST) for frame in proof_frames]
        images[0].save(output / "gameplay.gif", save_all=True, append_images=images[1:], duration=350, loop=0)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        episodes_reproducible = len(episode_hashes) == episodes and all(
            episode_hashes[0] == episode_hashes[index] for index in range(1, len(episode_hashes))
        )
        reset_hashes = [hashes[0] for hashes in episode_hashes]
        reset_reproducible = len(reset_hashes) == episodes and len(set(reset_hashes)) == 1
        first_reset_hash = reset_hashes[0] if reset_hashes else ""
        second_reset_hash = reset_hashes[1] if len(reset_hashes) > 1 else ""
        metrics = {
            "run_id": output.name,
            "checkpoint_loaded": bool(reset_info.get("checkpoint_loaded")),
            "checkpoint_run_id": checkpoint.parent.name,
            "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "reset_reproducible_observation": reset_reproducible,
            "first_reset_observation_sha256": first_reset_hash,
            "second_reset_observation_sha256": second_reset_hash,
            "observation_shape": list(obs.shape),
            "observation_dtype": str(obs.dtype),
            "action_names": list(GBA_ACTIONS),
            "requested_steps": steps * episodes,
            "steps_per_episode": steps,
            "completed_steps": len(rows),
            "episodes_requested": episodes,
            "episodes_completed": len(episode_hashes),
            "episodes_reproducible": episodes_reproducible,
            "action_counts": action_counts,
            "longest_identical_observation_streak": max(
                (longest_repeat_streak(hashes) for hashes in episode_hashes), default=0
            ),
            "action_frames": action_frames,
            "neutral_rewards_only": all(row["reward"] == 0.0 for row in rows),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "checkpoint_source_sha256_after_run": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_source_unchanged": checkpoint_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "limitations": [
                "The seeded random policy executes directional controls from a verified custom-party checkpoint; identical observations are a pixel-level diagnostic, not proof of a semantic stall.",
                "No battle/outcome rewards or PPO training are used.",
            ],
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        report = [
            "# GBA Gymnasium action smoke", "",
            f"Run {output.name} loaded a native checkpoint and executed {len(rows)} discrete-control steps.",
            "", f"- Reset observation reproducible: **{reset_reproducible}**",
            f"- Observation: {obs.shape} {obs.dtype}",
            "- Rewards: neutral (0.0) for every step",
            f"- Episode end: terminated={terminated}, truncated={truncated}",
            f"- Episodes reproducible: {episodes_reproducible} ({len(episode_hashes)}/{episodes})",
            f"- Action counts: {action_counts}",
            f"- Longest identical-observation streak: {metrics['longest_identical_observation_streak']} (not a semantic stuck detector)",
            "- Movement/progression semantics: not inferred by this smoke.",
            "", "The action trace, compact contact sheet, GIF, and metrics are local and ignored by Git.",
        ]
        (output / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        public = package_public_evidence(output)
        print(f"Sanitized evidence: {public}")
    finally:
        env.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="native Stable-Retro checkpoint; defaults to latest locally verified Seed checkpoint")
    parser.add_argument("--steps", type=int, default=12, help="actions per episode")
    parser.add_argument("--episodes", type=int, default=2, help="repeat the same seeded policy from checkpoint")
    parser.add_argument("--action-frames", type=int, default=8)
    parser.add_argument("--max-steps", type=int, help="episode truncation boundary; defaults to --steps")
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 2 or args.action_frames < 1 or (args.max_steps is not None and args.max_steps < args.steps):
        parser.error("steps and action-frames must be positive; episodes must be at least two; max-steps must cover --steps")
    print(run(args.checkpoint, args.steps, args.action_frames, args.max_steps, args.episodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
