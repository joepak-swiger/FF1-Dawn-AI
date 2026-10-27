"""Analyze one or more historical v0.9 and manual v1 RAM capture sessions."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ff1_dawn_ai.ram_analysis import DEFAULT_ADDRESSES, analyze_samples, discover_sessions, write_analysis


def parse_addresses(values):
    if not values:
        return DEFAULT_ADDRESSES
    try:
        return tuple(dict.fromkeys(int(value, 0) for value in values))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid RAM address: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Capture directory, capture file, or parent directory to scan")
    parser.add_argument("--address", action="append", help="RAM address to analyze (hex like 0x0014); repeat as needed")
    parser.add_argument("--output-dir", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args()

    try:
        addresses = parse_addresses(args.address)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    samples = discover_sessions(args.inputs, addresses=addresses)
    result = analyze_samples(samples, addresses=addresses)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    output_dir = args.output_dir or PROJECT_ROOT / "videos" / "nes_ram_analysis" / run_id
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    write_analysis(result, output_dir)
    print(f"Sessions analyzed: {result['session_count']}")
    print(f"Samples analyzed: {result['sample_count']}")
    print(f"Report: {output_dir / 'run_report.txt'}")
    print(f"Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
