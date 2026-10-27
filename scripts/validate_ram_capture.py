"""Fail-closed validation for one automated RAM capture run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def validate_run(run_dir: Path, expected_episodes: int = 3) -> list[str]:
    """Raise ValueError unless all sessions and paired capture files are complete."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError(f"Missing run metrics: {metrics_path}")
    run = json.loads(metrics_path.read_text(encoding="utf-8"))
    status = run.get("status")
    legacy_without_status = status is None
    if status not in ("complete", None):
        raise ValueError(f"Run status is {run.get('status')!r}, expected 'complete'")
    episodes = run.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episode records")
    ids = [item.get("session_id") for item in episodes]
    if len(set(ids)) != expected_episodes or any(not item for item in ids):
        raise ValueError("Episode records must have unique non-empty session IDs")

    for session_id in ids:
        session_dir = run_dir / session_id
        session_metrics_path = session_dir / "metrics.json"
        npz_path = session_dir / "ram_samples.npz"
        csv_path = session_dir / "probe_events.csv"
        if not all(path.is_file() for path in (session_metrics_path, npz_path, csv_path)):
            raise ValueError(f"{session_id} is missing metrics, RAM samples, or events CSV")
        session = json.loads(session_metrics_path.read_text(encoding="utf-8"))
        session_status = session.get("status")
        if session_status != "complete" and not (legacy_without_status and session_status is None):
            raise ValueError(f"{session_id} status is {session.get('status')!r}")
        steps = session.get("steps_recorded", 0)
        if not isinstance(steps, int) or steps < 2:
            raise ValueError(f"{session_id} has too few recorded samples")
        with np.load(npz_path, allow_pickle=False) as bundle:
            if not {"rams", "battle_like"}.issubset(bundle.files):
                raise ValueError(f"{session_id} has an unexpected NPZ schema")
            rams = bundle["rams"]
            labels = bundle["battle_like"]
            if rams.ndim != 2 or rams.shape[0] != steps or labels.size != steps:
                raise ValueError(f"{session_id} RAM sample count does not match its metrics")
            if rams.shape[1] <= 0:
                raise ValueError(f"{session_id} contains no RAM bytes")
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        if row_count != steps:
            raise ValueError(f"{session_id} CSV row count does not match its metrics")
        frames_dir = session_dir / "frames"
        if not frames_dir.is_dir() or not any(frames_dir.glob("*.png")):
            raise ValueError(f"{session_id} has no proof frames")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    try:
        session_ids = validate_run(args.run_dir, args.episodes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Capture validation failed: {exc}\n")
    print(f"Validated {len(session_ids)} complete sessions: {', '.join(session_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
