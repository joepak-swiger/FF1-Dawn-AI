"""Offline analysis for historical and manually labeled FF1 NES RAM captures.

Scene labels never come from the RAM values being evaluated. Historical v0.9
labels are the independent (but imperfect) pixel heuristic; v1 labels are the
user's manual labels. Scores are descriptive and are not proofs of semantics.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_ADDRESSES = (
    0x0014, 0x00D2, 0x00D3, 0x03C1, 0x03C2, 0x03C3, 0x03CD, 0x03CE,
    0x03CF, 0x03D1, 0x03D2, 0x03D3, 0x03D5, 0x03D9, 0x03DA, 0x03DB,
    0x03DD, 0x03DE, 0x03DF, 0x0270, 0x0271, 0x0272, 0x0273, 0x0274,
    0x0275, 0x00B5, 0x00D5, 0x00C5, 0x00D4, 0x00DC, 0x00DF,
)
RAM_COLUMN = re.compile(r"^ram_([0-9a-fA-F]{1,4})$")
FIELD_LABELS = {"field", "field_overworld"}
BATTLE_LABELS = {"battle"}
AMBIGUOUS_LABELS = {"", "unlabeled", "menu_town", "other", "unknown", "ambiguous"}


def pixel_scene_features(rgb: np.ndarray) -> dict:
    """Return the historical v0.9 pixel-only battle proxy, without reading RAM."""
    arr = np.asarray(rgb).astype(np.int16)
    gray = arr[:, :, :3].mean(axis=2)
    bottom = gray[int(gray.shape[0] * 0.55):, :]
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    dark_fraction = float(np.mean(gray < 40))
    bright_fraction = float(np.mean(gray > 185))
    bottom_bright_fraction = float(np.mean(bottom > 185))
    green_fraction = float(np.mean((green > red + 18) & (green > blue + 18) & (green > 65)))
    battle_like = (
        (dark_fraction > 0.28 and bottom_bright_fraction > 0.025 and green_fraction < 0.22)
        or (dark_fraction > 0.45 and bright_fraction > 0.018 and green_fraction < 0.28)
    )
    return {
        "battle_like": bool(battle_like),
        "dark_fraction": round(dark_fraction, 4),
        "bright_fraction": round(bright_fraction, 4),
        "bottom_bright_fraction": round(bottom_bright_fraction, 4),
        "green_fraction": round(green_fraction, 4),
    }


@dataclass(frozen=True)
class Sample:
    session_id: str
    step: int
    label: str
    label_source: str
    sample_type: str
    ram: dict[int, int]
    label_epoch: int | None = None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _ram_fields(row: dict[str, str]) -> dict[int, int]:
    values = {}
    for column, raw in row.items():
        match = RAM_COLUMN.match(column or "")
        if not match or raw in (None, ""):
            continue
        try:
            values[int(match.group(1), 16)] = int(raw)
        except (TypeError, ValueError):
            continue
    return values


def load_manual_csv(path: Path, session_id: str | None = None) -> list[Sample]:
    """Load v1 manually labeled action samples; exclude label-key snapshots."""
    samples = []
    sid = session_id or path.parent.name
    for row in _read_csv(path):
        if row.get("sample_type", "action") != "action":
            continue
        ram = _ram_fields(row)
        if not ram:
            continue
        try:
            step = int(row.get("step", 0))
            epoch = int(row["label_epoch"]) if row.get("label_epoch", "") not in (None, "") else None
        except (TypeError, ValueError):
            continue
        samples.append(Sample(
            session_id=sid, step=step,
            label=(row.get("label") or "unlabeled").strip().lower(),
            label_source="manual_v1", sample_type="action", ram=ram,
            label_epoch=epoch,
        ))
    if not samples:
        raise ValueError(f"No RAM action samples found in {path}")
    return samples


def load_v09_npz(
    path: Path,
    session_id: str | None = None,
    addresses: Iterable[int] = DEFAULT_ADDRESSES,
) -> list[Sample]:
    """Load v0.9 RAM matrix and independent image-heuristic labels."""
    events_path = path.with_name("probe_events.csv")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing paired v0.9 event CSV: {events_path}")
    with np.load(path, allow_pickle=False) as bundle:
        if not {"rams", "battle_like"}.issubset(bundle.files):
            raise ValueError(f"Unexpected v0.9 NPZ schema in {path}: {bundle.files}")
        rams = np.asarray(bundle["rams"], dtype=np.uint8)
        battle_like = np.asarray(bundle["battle_like"], dtype=bool).reshape(-1)
    if rams.ndim != 2 or rams.shape[0] != len(battle_like):
        raise ValueError(f"RAM/label dimensions do not match in {path}")
    events = _read_csv(events_path)
    if len(events) != len(battle_like):
        raise ValueError(f"NPZ/event row counts do not match in {path}")
    sid = session_id or path.parent.name
    addresses = tuple(addresses)
    return [Sample(
        session_id=sid,
        step=int(event.get("step", index)),
        label="battle" if is_battle_like else "field_overworld",
        label_source=event.get("label_source") or "visual_heuristic_v0_9",
        sample_type="action",
        ram={address: int(ram[address]) for address in addresses if address < len(ram)},
    ) for index, (ram, is_battle_like, event) in enumerate(zip(rams, battle_like, events))]


def discover_sessions(
    inputs: Iterable[str | Path],
    addresses: Iterable[int] = DEFAULT_ADDRESSES,
) -> list[Sample]:
    """Read one or more capture directories/files recursively, without duplicates."""
    manual_files: dict[Path, str] = {}
    npz_files: dict[Path, str] = {}
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            parent = candidate.parent
            session_id = (
                f"{parent.parent.name}/{parent.name}"
                if parent.name.startswith("session_") else parent.name
            )
            if candidate.name == "manual_ram_events.csv":
                manual_files[candidate] = session_id
            elif candidate.name == "ram_samples.npz":
                npz_files[candidate] = session_id
            elif candidate.name == "nes_ram_events.csv":
                manual_files[candidate] = session_id
    loaded: list[Sample] = []
    for path, session_id in sorted(npz_files.items()):
        loaded.extend(load_v09_npz(path, session_id, addresses))
    for path, session_id in sorted(manual_files.items()):
        loaded.extend(load_manual_csv(path, session_id))
    if not loaded:
        raise ValueError("No supported v0.9 NPZ or v1 manual RAM capture found")
    return loaded


def _distribution(values: list[int]) -> dict:
    counts = Counter(values)
    if not values:
        return {"count": 0, "mode": None, "mode_fraction": 0.0, "unique_values": [], "value_counts": {}}
    mode, mode_count = counts.most_common(1)[0]
    return {
        "count": len(values), "mode": mode,
        "mode_fraction": round(mode_count / len(values), 6),
        "unique_values": sorted(counts),
        "value_counts": {str(key): counts[key] for key in sorted(counts)},
    }


def _confusion(samples: list[Sample], address: int, field_mode: int | None, battle_mode: int | None) -> dict:
    if field_mode is None or battle_mode is None or field_mode == battle_mode:
        return {"true_positive": 0, "false_negative": 0, "false_positive": 0, "true_negative": 0,
                "labeled_count": 0, "false_positive_rate": None, "false_negative_rate": None}
    tp = fp = tn = fn = 0
    for sample in samples:
        value = sample.ram.get(address)
        if value is None or sample.label not in FIELD_LABELS | BATTLE_LABELS:
            continue
        actual = sample.label in BATTLE_LABELS
        predicted = value == battle_mode
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return {
        "true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
        "labeled_count": tp + fp + tn + fn,
        "false_positive_rate": round(fp / (fp + tn), 6) if fp + tn else None,
        "false_negative_rate": round(fn / (fn + tp), 6) if fn + tp else None,
    }


def analyze_samples(samples: Iterable[Sample], addresses: Iterable[int] = DEFAULT_ADDRESSES) -> dict:
    samples = sorted(samples, key=lambda sample: (sample.session_id, sample.step))
    session_ids = sorted({sample.session_id for sample in samples})
    candidate_rows = []
    transitions = []
    for address in addresses:
        field_samples = [s for s in samples if address in s.ram and s.label in FIELD_LABELS]
        battle_samples = [s for s in samples if address in s.ram and s.label in BATTLE_LABELS]
        field_dist = _distribution([s.ram[address] for s in field_samples])
        battle_dist = _distribution([s.ram[address] for s in battle_samples])
        field_mode, battle_mode = field_dist["mode"], battle_dist["mode"]
        separable = field_mode is not None and battle_mode is not None and field_mode != battle_mode

        per_source = {}
        for source in sorted({s.label_source for s in samples}):
            source_samples = [s for s in samples if s.label_source == source]
            per_source[source] = _confusion(source_samples, address, field_mode, battle_mode)
        per_session = []
        for sid in session_ids:
            local = [s for s in samples if s.session_id == sid]
            local_field = [s.ram[address] for s in local if address in s.ram and s.label in FIELD_LABELS]
            local_battle = [s.ram[address] for s in local if address in s.ram and s.label in BATTLE_LABELS]
            if local_field or local_battle:
                per_session.append({
                    "session_id": sid,
                    "field": _distribution(local_field),
                    "battle": _distribution(local_battle),
                    "confusion": _confusion(local, address, field_mode, battle_mode),
                })

        local_mode_pairs = [
            (item["field"]["mode"], item["battle"]["mode"])
            for item in per_session
            if item["field"]["count"] and item["battle"]["count"]
        ]
        same_distinct_modes_across_sessions = bool(local_mode_pairs) and all(
            field_value is not None and battle_value is not None and field_value != battle_value
            and (field_value, battle_value) == local_mode_pairs[0]
            for field_value, battle_value in local_mode_pairs
        )

        ambiguous = sum(address in s.ram and (s.label in AMBIGUOUS_LABELS or s.label not in FIELD_LABELS | BATTLE_LABELS) for s in samples)
        confusion_all = _confusion(samples, address, field_mode, battle_mode)
        positives = confusion_all["true_positive"] + confusion_all["false_negative"]
        negatives = confusion_all["true_negative"] + confusion_all["false_positive"]
        sensitivity = confusion_all["true_positive"] / positives if positives else None
        specificity = confusion_all["true_negative"] / negatives if negatives else None
        balanced_accuracy = round((sensitivity + specificity) / 2, 6) if sensitivity is not None and specificity is not None else None

        for sid in session_ids:
            local = sorted((s for s in samples if s.session_id == sid and address in s.ram), key=lambda s: s.step)
            for previous, current in zip(local, local[1:]):
                old, new = previous.ram[address], current.ram[address]
                if old == new:
                    continue
                if separable:
                    old_battle, new_battle = old == battle_mode, new == battle_mode
                    kind = "enter_battle_candidate" if new_battle and not old_battle else (
                        "exit_battle_candidate" if old_battle and not new_battle else "value_change"
                    )
                else:
                    kind = "value_change"
                label = current.label
                if label in AMBIGUOUS_LABELS or label not in FIELD_LABELS | BATTLE_LABELS:
                    status = "ambiguous"
                elif (kind == "enter_battle_candidate" and label in BATTLE_LABELS) or (
                    kind == "exit_battle_candidate" and label in FIELD_LABELS
                ) or kind == "value_change":
                    status = "agrees"
                else:
                    status = "label_conflict"
                transitions.append({
                    "session_id": sid, "label_source": current.label_source,
                    "address_hex": f"0x{address:04X}", "step": current.step,
                    "from_value": old, "to_value": new, "event": kind,
                    "label": label, "label_status": status, "label_epoch": current.label_epoch,
                })

        candidate_rows.append({
            "address_hex": f"0x{address:04X}", "address_dec": address,
            "field": field_dist, "battle": battle_dist,
            "mode_separable": separable,
            "confusion_if_battle_mode_is_battle": confusion_all,
            "balanced_accuracy": balanced_accuracy,
            "confusion_by_label_source": per_source,
            "ambiguous_or_non_binary_samples": ambiguous,
            "sessions": per_session,
            "sessions_with_both_labels": len(local_mode_pairs),
            "same_distinct_modes_across_sessions": same_distinct_modes_across_sessions,
            "transition_count": sum(t["address_hex"] == f"0x{address:04X}" for t in transitions),
        })

    candidate_rows.sort(key=lambda row: (
        not row["mode_separable"],
        row["balanced_accuracy"] is None,
        -(row["balanced_accuracy"] or 0),
        -row["confusion_if_battle_mode_is_battle"]["labeled_count"],
    ))
    transitions.sort(key=lambda row: (row["session_id"], row["address_hex"], row["step"]))
    return {
        "session_ids": session_ids, "session_count": len(session_ids),
        "sample_count": len(samples),
        "label_counts": dict(Counter(sample.label for sample in samples)),
        "label_source_counts": dict(Counter(sample.label_source for sample in samples)),
        "candidates": candidate_rows, "transitions": transitions,
        "interpretation": (
            "Distributions and disagreements describe these captures only. v0.9 labels are from pixels; "
            "v1 labels are manual but may persist through unobserved transitions. Candidate modes and edge "
            "counts do not establish the semantic meaning of a RAM byte."
        ),
    }


def write_analysis(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "candidate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["address_hex", "field_count", "field_mode", "field_mode_fraction", "battle_count", "battle_mode",
                  "battle_mode_fraction", "mode_separable", "false_positive", "false_negative", "true_positive",
                  "true_negative", "false_positive_rate", "false_negative_rate", "balanced_accuracy", "ambiguous_samples", "session_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["candidates"]:
            c = row["confusion_if_battle_mode_is_battle"]
            writer.writerow({
                "address_hex": row["address_hex"], "field_count": row["field"]["count"],
                "field_mode": row["field"]["mode"], "field_mode_fraction": row["field"]["mode_fraction"],
                "battle_count": row["battle"]["count"], "battle_mode": row["battle"]["mode"],
                "battle_mode_fraction": row["battle"]["mode_fraction"], "mode_separable": row["mode_separable"],
                "false_positive": c["false_positive"], "false_negative": c["false_negative"],
                "true_positive": c["true_positive"], "true_negative": c["true_negative"],
                "false_positive_rate": c["false_positive_rate"], "false_negative_rate": c["false_negative_rate"],
                "balanced_accuracy": row["balanced_accuracy"],
                "ambiguous_samples": row["ambiguous_or_non_binary_samples"], "session_count": len(row["sessions"]),
            })
    with (output_dir / "transitions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["session_id", "label_source", "address_hex", "step", "from_value", "to_value", "event", "label", "label_status", "label_epoch"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in result["transitions"])

    lines = ["FF1 NES RAM analysis", "====================",
             f"Sessions: {result['session_count']} ({', '.join(result['session_ids'])})",
             f"Samples: {result['sample_count']}", f"Labels: {result['label_counts']}",
             f"Label sources: {result['label_source_counts']}", "",
             "Candidates (pooled label-conditioned battle mode used as a candidate value)", "-" * 80]
    for row in result["candidates"]:
        c = row["confusion_if_battle_mode_is_battle"]
        lines.append(
            f"{row['address_hex']} field={row['field']['mode']} ({row['field']['mode_fraction']:.3f}, n={row['field']['count']}) "
            f"battle={row['battle']['mode']} ({row['battle']['mode_fraction']:.3f}, n={row['battle']['count']}) "
            f"BA={row['balanced_accuracy']} FP={c['false_positive']} FN={c['false_negative']} "
            f"ambiguous={row['ambiguous_or_non_binary_samples']} "
            f"sessions_with_both_labels={row['sessions_with_both_labels']} "
            f"same_distinct_modes_across_sessions={row['same_distinct_modes_across_sessions']}"
        )
    conflict_count = sum(row["label_status"] == "label_conflict" for row in result["transitions"])
    lines.extend(["", f"RAM transitions: {len(result['transitions'])}",
                  f"Transitions conflicting with the current binary label: {conflict_count}",
                  "", "Interpretation", "--------------", result["interpretation"], "",
                  "FP/FN compare the modal-value candidate rule with assigned binary labels; they are not manually adjudicated ground-truth errors.",
                  "See confusion_by_label_source in metrics.json to separate pixel-proxy and manual-label disagreements.",
                  "",
                  "Outputs: metrics.json, candidate_summary.csv, transitions.csv"])
    (output_dir / "run_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
