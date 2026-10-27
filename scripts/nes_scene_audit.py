"""Extract compact, RAM-independent review evidence from an NES auto-capture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ff1_dawn_ai.nes_scene_audit import parse_anchor, run_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="completed videos/nes_ram_auto_capture/<run-id> directory")
    parser.add_argument("output", type=Path, help="new evidence directory; existing output is never overwritten")
    parser.add_argument("--anchor", action="append", default=[], metavar="SESSION:STEP=LABEL",
                        help="independently reviewed screenshot reference; may be repeated")
    args = parser.parse_args()
    capture = args.capture if args.capture.is_absolute() else PROJECT_ROOT / args.capture
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    try:
        result = run_audit(capture, output, [parse_anchor(value) for value in args.anchor])
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Scene audit failed: {exc}\n")
    print(f"Scene audit report: {result / 'report.md'}")
    print(f"Metrics: {result / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
