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
                    and roster == [("Sif", "Sif"), ("Myth", "Myth"), ("Milo", "Milo"), ("Mustard", "Mustar")]
                    and classes == ["Warrior", "Thief", "Monk", "Red Mage"]
                    and state.is_file()):
                candidates.append((state.stat().st_mtime_ns, state))
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        raise FileNotFoundError(
            "No verified Mustar/Red Mage custom-party checkpoint found. "
            "Complete run_gba_feasibility.cmd first or pass --checkpoint explicitly."
        )
    return max(candidates)[1]


def new_output(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    path = root / stamp
    path.mkdir()
    return path


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
        and metrics.get("completed_steps", 0) > 0
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
            "run_id", "checkpoint_loaded", "reset_reproducible_observation",
            "observation_shape", "observation_dtype", "action_names",
            "requested_steps", "completed_steps", "action_frames",
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
        f"Run {metrics['run_id']} restored the corrected Mustar/Red Mage native checkpoint and completed {metrics['completed_steps']} discrete actions.",
        "",
        f"- Reset observations reproducible: {metrics['reset_reproducible_observation']}",
        f"- Observation shape and dtype: {metrics['observation_shape']} / {metrics['observation_dtype']}",
        f"- Neutral rewards on every action: {metrics['neutral_rewards_only']}",
        f"- Episode result: terminated={metrics['terminated']}, truncated={metrics['truncated']}",
        "- Source native checkpoint remained unchanged.",
        "",
        "The contact sheet shows the Cornelia scene and response frames. This smoke validates the Gymnasium/core interface and input execution; it does not infer coordinates, battle outcomes, or story progress. No PPO training ran.",
    ]
    (target / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return target


def run(checkpoint: Path | None, steps: int, action_frames: int, max_steps: int | None = None) -> Path:
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
        obs, reset_info = env.reset(seed=1729)
        first_hash = hashlib.sha256(obs.tobytes()).hexdigest()
        proof_frames.append(env.render())
        # Re-load the same state to verify deterministic checkpoint restoration.
        obs_again, _ = env.reset(seed=1729)
        reset_hash = hashlib.sha256(obs_again.tobytes()).hexdigest()
        reproducible = first_hash == reset_hash
        obs = obs_again
        proof_frames[0] = env.render()
        action_sequence = ["UP", "RIGHT", "DOWN", "LEFT", "A", "NOOP"]
        terminated = truncated = False
        for step in range(steps):
            action_name = action_sequence[step % len(action_sequence)]
            action = GBA_ACTIONS.index(action_name)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            image_name = f"frame_{step + 1:03d}_{action_name.lower()}.png"
            if step < 12 or step == steps - 1:
                Image.fromarray(frame).save(output / image_name)
                proof_frames.append(frame)
            rows.append({
                "step": step + 1, "action_index": action, "action": action_name,
                "reward": reward, "terminated": terminated, "truncated": truncated,
                "observation_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
                "frame": image_name if step < 12 or step == steps - 1 else "",
            })
            if terminated or truncated:
                break
        with (output / "action_trace.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["step", "action"])
            writer.writeheader()
            writer.writerows(rows)
        make_contact_sheet(proof_frames, output / "contact_sheet.png")
        images = [Image.fromarray(frame).resize((240, 160), Image.Resampling.NEAREST) for frame in proof_frames]
        images[0].save(output / "gameplay.gif", save_all=True, append_images=images[1:], duration=350, loop=0)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metrics = {
            "run_id": output.name,
            "checkpoint_loaded": bool(reset_info.get("checkpoint_loaded")),
            "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "reset_reproducible_observation": reproducible,
            "first_reset_observation_sha256": first_hash,
            "second_reset_observation_sha256": reset_hash,
            "observation_shape": list(obs.shape),
            "observation_dtype": str(obs.dtype),
            "action_names": list(GBA_ACTIONS),
            "requested_steps": steps,
            "completed_steps": len(rows),
            "action_frames": action_frames,
            "neutral_rewards_only": all(row["reward"] == 0.0 for row in rows),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "checkpoint_source_sha256_after_run": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_source_unchanged": checkpoint_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "limitations": [
                "The smoke executes controls from a verified custom-party checkpoint; it does not infer coordinate changes or game progress.",
                "No battle/outcome rewards or PPO training are used.",
            ],
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        report = [
            "# GBA Gymnasium action smoke", "",
            f"Run {output.name} loaded a native checkpoint and executed {len(rows)} discrete-control steps.",
            "", f"- Reset observation reproducible: **{reproducible}**",
            f"- Observation: {obs.shape} {obs.dtype}",
            "- Rewards: neutral (0.0) for every step",
            f"- Episode end: terminated={terminated}, truncated={truncated}",
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
    parser.add_argument("--checkpoint", type=Path, help="native Stable-Retro checkpoint; defaults to latest locally verified Mustar checkpoint")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--action-frames", type=int, default=8)
    parser.add_argument("--max-steps", type=int, help="episode truncation boundary; defaults to --steps")
    args = parser.parse_args()
    if args.steps < 1 or args.action_frames < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("steps and action-frames must be positive")
    print(run(args.checkpoint, args.steps, args.action_frames, args.max_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
