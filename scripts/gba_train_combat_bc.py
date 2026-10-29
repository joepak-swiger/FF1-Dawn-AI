#!/usr/bin/env python3
"""Train a local supervised GBA combat policy from real expert decisions."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ff1_dawn_ai.gba_combat_learning import train_from_directories


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True,
                        help="one or more local expert_decisions capture directories")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group-map", type=Path,
                        help="optional JSON mapping replay-state hashes to original encounter-parent hashes")
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.group_map.read_text(encoding="utf-8")) if args.group_map else None
        if mapping is not None and (not isinstance(mapping, dict) or
                                    any(not isinstance(k, str) or not isinstance(v, str)
                                        for k, v in mapping.items())):
            raise ValueError("--group-map must contain a JSON object of string hash mappings")
        metrics = train_from_directories(args.dataset, args.model, seed=args.seed,
                                         group_mapping=mapping)
    except (ValueError, FileNotFoundError) as exc:
        print(f"TRAINING BLOCKED: {exc}", file=sys.stderr)
        return 2
    report = args.model.with_suffix(".metrics.json")
    report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Trained model: {args.model}\nEvaluation: {report}\n" + json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
