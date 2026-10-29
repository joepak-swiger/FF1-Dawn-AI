#!/usr/bin/env python3
"""Train a local supervised GBA combat policy from real expert decisions."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ff1_dawn_ai.gba_combat_learning import train_from_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        metrics = train_from_directory(args.dataset, args.model, seed=args.seed)
    except (ValueError, FileNotFoundError) as exc:
        print(f"TRAINING BLOCKED: {exc}", file=sys.stderr)
        return 2
    report = args.model.with_suffix(".metrics.json")
    report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Trained model: {args.model}\nEvaluation: {report}\n" + json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
