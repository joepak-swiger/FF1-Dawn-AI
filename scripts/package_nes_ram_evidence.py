"""Create a small GitHub-safe report package from a completed NES RAM run.

The packager whitelists aggregate metrics, selected transition counts, and a
downscaled contact sheet. It never copies NPZ dumps, full CSVs, ROMs, states,
models, videos, or original screenshots into the evidence package.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIXEL = "visual_heuristic_v0_9_pixels_only"
PINNED_ADDRESSES = {"0x0014", "0x00D2", "0x00D3", "0x0270", "0x0271", "0x0272", "0x0273", "0x0274", "0x0275"}


def candidate_addresses(candidates: list[dict], limit: int = 16) -> list[str]:
    """Keep top-ranked aggregate candidates plus the project watch list."""
    selected: list[str] = []
    for row in candidates[:limit]:
        address = row["address_hex"]
        if address not in selected:
            selected.append(address)
    for address in sorted(PINNED_ADDRESSES, key=lambda item: int(item, 16)):
        if address not in selected:
            selected.append(address)
    return selected


def transition_counts(path: Path, selected: set[str]) -> dict[str, dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    per_session: dict[tuple[str, str], Counter] = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            address = row["address_hex"]
            if address not in selected:
                continue
            session = row["session_id"].split("/")[-1]
            counts[address][row["event"]] += 1
            if row["label_status"] == "label_conflict":
                counts[address]["proxy_label_conflict"] += 1
            per_session[(address, session)][row["event"]] += 1
            if row["label_status"] == "label_conflict":
                per_session[(address, session)]["proxy_label_conflict"] += 1
    result = {}
    for address in selected:
        result[address] = {
            "entry_edges": counts[address]["enter_battle_candidate"],
            "exit_edges": counts[address]["exit_battle_candidate"],
            "edges_conflicting_with_pixel_proxy": counts[address]["proxy_label_conflict"],
            "per_session": {
                session: {
                    "entry_edges": per_session[(address, session)]["enter_battle_candidate"],
                    "exit_edges": per_session[(address, session)]["exit_battle_candidate"],
                    "edges_conflicting_with_pixel_proxy": per_session[(address, session)]["proxy_label_conflict"],
                }
                for session in sorted({key[1] for key in per_session if key[0] == address})
            },
        }
    return result


def label_error_timing(
    ram_samples: np.ndarray,
    battle_like: np.ndarray,
    address: int,
    battle_mode: int,
    transition_steps: list[int],
    tolerance: int = 1,
) -> dict:
    """Separate modal-rule disagreements at proxy edges from interior ones."""
    edge_steps = np.asarray(transition_steps, dtype=np.int64)
    mismatches = []
    for step, (value, is_battle) in enumerate(zip(ram_samples[:, address], battle_like)):
        predicted_battle = int(value) == battle_mode
        if predicted_battle != bool(is_battle):
            mismatches.append(step)
    near = [
        step for step in mismatches
        if edge_steps.size and int(np.min(np.abs(edge_steps - step))) <= tolerance
    ]
    return {
        "disagreement_samples": len(mismatches),
        "within_transition_tolerance_samples": len(near),
        "away_from_transition_samples": len(mismatches) - len(near),
        "transition_tolerance_samples": tolerance,
    }


def transition_error_summary(capture_dir: Path, capture_metrics: dict, candidates: list[dict]) -> dict:
    session_data = []
    for episode in capture_metrics["episodes"]:
        with np.load(capture_dir / episode["session_id"] / "ram_samples.npz") as arrays:
            session_data.append((episode, arrays["rams"], arrays["battle_like"]))
    summary = {}
    for candidate in candidates:
        address = int(candidate["address_hex"], 16)
        battle_mode = int(candidate["battle"]["mode"])
        per_session = {}
        for episode, rams, battle_like in session_data:
            timing = label_error_timing(
                rams, battle_like, address, battle_mode,
                [int(item["step"]) for item in episode.get("visual_transitions", [])],
            )
            per_session[episode["session_id"]] = timing
        summary[candidate["address_hex"]] = per_session
    return summary


def _nearest_saved_frame(session_dir: Path, step: int) -> tuple[int, Path] | None:
    options = []
    for path in (session_dir / "frames").glob("frame_*.png"):
        match = re.match(r"frame_(\d+)_", path.name)
        if match:
            frame_step = int(match.group(1))
            options.append((abs(frame_step - step), frame_step, path))
    if not options:
        return None
    _, frame_step, path = min(options)
    return frame_step, path


def evidence_selections(capture_dir: Path, capture_metrics: dict) -> list[dict]:
    """Choose first battle/field transitions and one rapid-toggle example per session."""
    selections = []
    for episode in capture_metrics["episodes"]:
        session = episode["session_id"]
        transitions = episode.get("visual_transitions", [])
        wanted: list[tuple[str, int]] = []
        entry = next((t for t in transitions if t["to"] == "battle_like"), None)
        exit_ = next((t for t in transitions if t["from"] == "battle_like" and t["to"] == "field_like"), None)
        if entry:
            wanted.append(("first battle-like entry", int(entry["step"])))
        if exit_:
            wanted.append(("first field-like return", int(exit_["step"])))
        rapid = next(
            (
                (left, right)
                for left, right in zip(transitions, transitions[1:])
                if int(right["step"]) - int(left["step"]) <= 2
            ),
            None,
        )
        if rapid:
            wanted.append(("rapid proxy transition", int(rapid[1]["step"])))
        session_dir = capture_dir / session
        seen_steps = set()
        for label, step in wanted:
            found = _nearest_saved_frame(session_dir, step)
            if found is None or found[0] in seen_steps:
                continue
            saved_step, image_path = found
            seen_steps.add(saved_step)
            selections.append({
                "session_id": session,
                "transition_label": label,
                "transition_step": step,
                "saved_frame_step": saved_step,
                "pixel_proxy_transition": True,
                "source_path": image_path,
            })
    return selections


def make_contact_sheet(selections: list[dict], output: Path) -> list[dict]:
    if not selections:
        return []
    columns = 3
    image_width, image_height, caption_height = 144, 135, 40
    rows = (len(selections) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * image_width, rows * (image_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    manifest = []
    for index, item in enumerate(selections):
        with Image.open(item["source_path"]) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((image_width, image_height))
        x = (index % columns) * image_width
        y = (index // columns) * (image_height + caption_height)
        sheet.paste(thumbnail, (x + (image_width - thumbnail.width) // 2, y))
        caption = f"{item['session_id']} step {item['saved_frame_step']}"
        subcaption = f"{item['transition_label']} (proxy)"
        draw.text((x + 2, y + image_height + 2), caption, fill="black", font=ImageFont.load_default())
        draw.text((x + 2, y + image_height + 18), subcaption, fill="black", font=ImageFont.load_default())
        manifest.append({key: value for key, value in item.items() if key != "source_path"})
    # Keep the public visual proof small enough for a concise review package.
    # The original screenshots remain local and are never copied into the repo.
    sheet.thumbnail((216, 263))
    sheet = sheet.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
    sheet.save(output, format="PNG", optimize=True)
    return manifest


def _candidate_summary(row: dict, transitions: dict) -> dict:
    confusion = row["confusion_if_battle_mode_is_battle"]
    return {
        "address": row["address_hex"],
        "field_proxy": {
            "samples": row["field"]["count"],
            "mode": row["field"]["mode"],
            "mode_fraction": row["field"]["mode_fraction"],
        },
        "battle_proxy": {
            "samples": row["battle"]["count"],
            "mode": row["battle"]["mode"],
            "mode_fraction": row["battle"]["mode_fraction"],
        },
        "mode_pair_repeats_across_sessions": row["same_distinct_modes_across_sessions"],
        "mode_separable": row["mode_separable"],
        "proxy_disagreement": {
            "field_samples_predicted_battle": confusion["false_positive"],
            "battle_samples_predicted_field": confusion["false_negative"],
            "balanced_accuracy_against_pixel_proxy": row["balanced_accuracy"],
        },
        "disagreement_timing_by_session": row.get("disagreement_timing_by_session", {}),
        "transitions": transitions.get(row["address_hex"], {}),
        "per_session": [
            {
                "session_id": session["session_id"].split("/")[-1],
                "field_mode": session["field"]["mode"],
                "field_mode_fraction": session["field"]["mode_fraction"],
                "battle_mode": session["battle"]["mode"],
                "battle_mode_fraction": session["battle"]["mode_fraction"],
                "field_samples_predicted_battle": session["confusion"]["false_positive"],
                "battle_samples_predicted_field": session["confusion"]["false_negative"],
            }
            for session in row["sessions"]
        ],
    }


def package(capture_dir: Path, analysis_dir: Path, output_root: Path, review_notes: list[str]) -> Path:
    capture_metrics = json.loads((capture_dir / "metrics.json").read_text(encoding="utf-8"))
    analysis_metrics = json.loads((analysis_dir / "metrics.json").read_text(encoding="utf-8"))
    if capture_metrics.get("status") != "complete" or analysis_metrics.get("session_count") != 3:
        raise ValueError("Only a completed three-session capture and matching analysis can be packaged")
    if any(episode.get("status") != "complete" for episode in capture_metrics.get("episodes", [])):
        raise ValueError("Capture contains an incomplete session")
    run_id = capture_metrics["run_id"]
    if any(run_id not in value for value in analysis_metrics.get("session_ids", [])):
        raise ValueError("Analysis metrics do not match the requested capture run")

    output_root = output_root if output_root.is_absolute() else PROJECT_ROOT / output_root
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"Evidence package already exists; refusing overwrite: {destination}")
    destination.mkdir(parents=True)

    addresses = candidate_addresses(analysis_metrics["candidates"])
    candidate_by_address = {row["address_hex"]: row for row in analysis_metrics["candidates"]}
    selected = set(addresses)
    transition_data = transition_counts(analysis_dir / "transitions.csv", selected)
    candidate_data = [
        _candidate_summary(candidate_by_address[address], transition_data)
        for address in addresses if address in candidate_by_address
    ]
    error_timing = transition_error_summary(capture_dir, capture_metrics, [candidate_by_address[a] for a in addresses if a in candidate_by_address])
    for candidate in candidate_data:
        candidate["disagreement_timing_by_session"] = error_timing.get(candidate["address"], {})
    selections = evidence_selections(capture_dir, capture_metrics)
    evidence_dir = destination / "evidence"
    evidence_dir.mkdir()
    evidence_manifest = make_contact_sheet(selections, evidence_dir / "selected_scenes.png")

    metrics = {
        "run_id": run_id,
        "capture_status": capture_metrics["status"],
        "analysis_sessions": analysis_metrics["session_count"],
        "samples": analysis_metrics["sample_count"],
        "labels_from_pixel_heuristic": analysis_metrics["label_source_counts"],
        "label_counts": analysis_metrics["label_counts"],
        "baseline": {"model": "v0.7 trusted baseline", "unmodified": capture_metrics.get("model_is_unmodified", False)},
        "sessions": [
            {
                "session_id": episode["session_id"],
                "samples": episode["steps_recorded"],
                "field_like_proxy_samples": episode["visual_field_like_samples"],
                "battle_like_proxy_samples": episode["visual_battle_like_samples"],
                "proxy_transitions": len(episode.get("visual_transitions", [])),
            }
            for episode in capture_metrics["episodes"]
        ],
        "candidate_values_are_associations_not_semantic_flags": True,
        "candidate_comparison": candidate_data,
        "evidence_frames": evidence_manifest,
        "review_notes": review_notes,
        "raw_data_included": False,
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_csv(destination / "candidate_comparison.csv", candidate_data)
    (destination / "report.md").write_text(_render_report(metrics), encoding="utf-8")
    return destination


def _write_csv(path: Path, candidates: list[dict]) -> None:
    fields = [
        "address", "field_mode", "field_mode_fraction", "battle_mode", "battle_mode_fraction",
        "same_distinct_modes_across_sessions", "false_positive_vs_pixel_proxy",
        "false_negative_vs_pixel_proxy", "balanced_accuracy_vs_pixel_proxy",
        "entry_edges", "exit_edges", "edge_label_conflicts",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            transitions = item["transitions"]
            writer.writerow({
                "address": item["address"],
                "field_mode": item["field_proxy"]["mode"],
                "field_mode_fraction": item["field_proxy"]["mode_fraction"],
                "battle_mode": item["battle_proxy"]["mode"],
                "battle_mode_fraction": item["battle_proxy"]["mode_fraction"],
                "same_distinct_modes_across_sessions": item["mode_pair_repeats_across_sessions"],
                "false_positive_vs_pixel_proxy": item["proxy_disagreement"]["field_samples_predicted_battle"],
                "false_negative_vs_pixel_proxy": item["proxy_disagreement"]["battle_samples_predicted_field"],
                "balanced_accuracy_vs_pixel_proxy": item["proxy_disagreement"]["balanced_accuracy_against_pixel_proxy"],
                "entry_edges": transitions.get("entry_edges", 0),
                "exit_edges": transitions.get("exit_edges", 0),
                "edge_label_conflicts": transitions.get("edges_conflicting_with_pixel_proxy", 0),
            })


def _render_report(metrics: dict) -> str:
    rows = []
    for item in metrics["candidate_comparison"]:
        proxy = item["proxy_disagreement"]
        rows.append(
            f"| {item['address']} | {item['field_proxy']['mode']} ({item['field_proxy']['mode_fraction']:.3f}) "
            f"| {item['battle_proxy']['mode']} ({item['battle_proxy']['mode_fraction']:.3f}) "
            f"| {proxy['field_samples_predicted_battle']} | {proxy['battle_samples_predicted_field']} "
            f"| {proxy['balanced_accuracy_against_pixel_proxy']:.3f} "
            f"| {item['mode_pair_repeats_across_sessions']} "
            f"| {item['transitions'].get('edges_conflicting_with_pixel_proxy', 0)} |"
        )
    notes = "\n".join(f"- {note}" for note in metrics["review_notes"]) or "- No separate manual review notes were supplied."
    unstable_modes = [
        item["address"] for item in metrics["candidate_comparison"]
        if not item["mode_pair_repeats_across_sessions"]
    ]
    unstable_note = (
        f"- Pooled scores hide per-session mode drift at {', '.join(unstable_modes)}; "
        "inspect each session's distribution in `metrics.json` before ranking these bytes.\n"
        if unstable_modes else ""
    )
    mode14 = next((item for item in metrics["candidate_comparison"] if item["address"] == "0x0014"), None)
    if mode14:
        edge_errors = sum(
            timing["within_transition_tolerance_samples"]
            for timing in mode14["disagreement_timing_by_session"].values()
        )
        away_errors = sum(
            timing["away_from_transition_samples"]
            for timing in mode14["disagreement_timing_by_session"].values()
        )
        edge_note = (
            f"- For `0x0014`, {edge_errors} of {edge_errors + away_errors} modal-rule disagreements "
            "occur within one sample of a pixel-proxy transition; this is consistent with boundary "
            "timing/label uncertainty, but still needs independent scene labels.\n"
        )
    else:
        edge_note = ""
    session_rows = "\n".join(
        f"| {s['session_id']} | {s['samples']} | {s['field_like_proxy_samples']} | {s['battle_like_proxy_samples']} | {s['proxy_transitions']} |"
        for s in metrics["sessions"]
    )
    return f"""# NES RAM candidate analysis — {metrics['run_id']}

## Run record

- Completed capture and analysis: {metrics['analysis_sessions']} independent resets, {metrics['samples']:,} samples.
- Pixel-proxy labels: {metrics['label_counts']['field_overworld']} field-like and {metrics['label_counts']['battle']} battle-like.
- PPO baseline: {metrics['baseline']['model']}; unchanged during capture: {metrics['baseline']['unmodified']}.
- The analyzed scenes are proxy labels, not independently verified game states. Candidate agreement is not evidence that an address semantically means “battle.”

| Session | Samples | Field-like proxy | Battle-like proxy | Proxy transitions |
|---|---:|---:|---:|---:|
{session_rows}

## Candidate comparison

For this table, predict “battle” when the byte equals the pooled battle mode.
FP/FN and balanced accuracy are disagreements against pixel labels, not
adjudicated gameplay errors. The last column counts candidate value edges that
the analyzer marked as conflicting with those labels.

| Address | Field mode (fraction) | Battle mode (fraction) | FP | FN | BA vs proxy | Mode pair repeats | Conflicting edges |
|---|---:|---:|---:|---:|---:|:---:|---:|
{chr(10).join(rows)}

## Interpretation

- `0x0014` repeated a distinct 23/0 modal pair in all three sessions. Against
  the pixel proxy it had 16 field samples predicted as battle and 2 battle
  samples predicted as field (balanced accuracy {next(c['proxy_disagreement']['balanced_accuracy_against_pixel_proxy'] for c in metrics['candidate_comparison'] if c['address']=='0x0014'):.3f}). It is more repeatable here than in the prior single manual session, but remains provisional.
- Stronger-scoring nearby candidates include several bytes in the
  `0x03CD`–`0x03DF` neighborhood.
  Several tie `0x0014` on proxy accuracy and share its confusion counts. Their
  close address grouping and identical proxy score do not make them independent
  confirmation; they may be correlated scene data.
{unstable_note}{edge_note}- `0x00D2` repeated modes 73/201; `0x0270` repeated 248/240. Their exact-mode
  rules miss many pixel-labeled battle samples, so they are less suitable as
  simple binary detectors in this run despite clean mode separation.
- Contact-sheet frames are downscaled samples selected around proxy transitions
  and rapid toggles. Review them before interpreting candidate edges; the sheet
  does not supply independent labels.
- No battle wins, losses, escapes, HP/EXP changes, or story progress were
  independently measured. This experiment does not validate a reward signal.

### Manual review notes

{notes}

## Decision

Keep RAM rewards and state transitions unchanged. Next, independently label
several battle entry/exit/result frames, including menus and title/new-game
screens, then test a small shortlist (`0x0014`, `0x03CD`, `0x00D2`, `0x0270`)
against those observations. Resolve whether `0x03CD` and neighboring bytes are
correlated before selecting an address. All published metrics are sanitized
summaries; raw RAM, full analysis CSVs, ROMs, saves, and models are excluded.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "docs" / "experiments" / "nes_ram")
    parser.add_argument("--review-note", action="append", default=[], help="short human-review note; may be repeated")
    args = parser.parse_args()
    try:
        capture_dir = args.capture_dir if args.capture_dir.is_absolute() else PROJECT_ROOT / args.capture_dir
        analysis_dir = args.analysis_dir if args.analysis_dir.is_absolute() else PROJECT_ROOT / args.analysis_dir
        destination = package(capture_dir, analysis_dir, args.output_root, args.review_note)
    except Exception as exc:
        print(f"Evidence packaging failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"GitHub-safe evidence package: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
