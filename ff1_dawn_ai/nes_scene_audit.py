"""Pixel-only layout triage and targeted evidence extraction for NES captures.

This module does not read emulator RAM. Its coarse visual families are review
prioritizers, not gameplay-state labels or reward signals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visual_layout_features(rgb: np.ndarray) -> dict[str, float]:
    """Measure coarse, RAM-independent color/layout cues on an RGB screenshot."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an HxWxRGB observation, got {image.shape}")
    image = image[:, :, :3].astype(np.int16)
    height, width = image.shape[:2]
    if height < 80 or width < 100:
        raise ValueError(f"Observation is too small for NES layout triage: {width}x{height}")
    gray = image.mean(axis=2)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    green_mask = (green > red + 18) & (green > blue + 18) & (green > 65)
    far_right = gray[int(height * 0.06):int(height * 0.74), int(width * 0.91):int(width * 0.96)]
    center = gray[int(height * 0.25):int(height * 0.72), int(width * 0.2):int(width * 0.8)]
    return {
        "green_fraction": float(green_mask.mean()),
        "dark_fraction": float((gray < 40).mean()),
        "bright_fraction": float((gray > 185).mean()),
        "right_party_panel_edge_brightness": float((far_right > 185).mean()),
        "center_menu_region_brightness": float((center > 185).mean()),
    }


def visual_layout_family(features: dict[str, float]) -> str:
    """Return a coarse screen-layout hypothesis; intentionally abstains broadly."""
    if features["green_fraction"] >= 0.20 and features["dark_fraction"] < 0.55:
        return "overworld_map_layout_candidate"
    if (
        features["right_party_panel_edge_brightness"] > 0.10
        and features["green_fraction"] < 0.10
    ):
        return "battle_party_panel_layout_candidate"
    if (
        features["dark_fraction"] >= 0.70
        and features["green_fraction"] < 0.10
        and features["right_party_panel_edge_brightness"] < 0.05
    ):
        return "dark_menu_or_transition_layout_candidate"
    return "unclassified_layout"


def _frames(session_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in (session_dir / "frames").glob("frame_*.png"):
        match = re.match(r"frame_(\d+)_", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _nearest_frame(frames: list[tuple[int, Path]], step: int) -> tuple[int, Path] | None:
    return min(frames, key=lambda item: (abs(item[0] - step), item[0])) if frames else None


def select_review_frames(capture_dir: Path, capture_metrics: dict, anchors: list[dict]) -> list[dict]:
    """Pick frames around pixel-proxy transitions plus explicitly reviewed anchors."""
    selected: dict[tuple[str, int], dict] = {}
    for session in capture_metrics.get("episodes", []):
        session_id = session["session_id"]
        frames = _frames(capture_dir / session_id)
        if not frames:
            continue
        with (capture_dir / session_id / "probe_events.csv").open("r", newline="", encoding="utf-8-sig") as handle:
            events = list(csv.DictReader(handle))
        proxy_by_step = {
            int(row["step"]): str(row.get("battle_like", "false")).strip().lower() in {"1", "true", "yes"}
            for row in events
        }
        steps = [int(frame_step) for frame_step, _ in frames]

        def add(frame_step: int, path: Path, reason: str, transition_step: int | None, review_label: str | None = None):
            key = (session_id, frame_step)
            item = selected.setdefault(key, {
                "session_id": session_id,
                "frame_step": frame_step,
                "image_path": path,
                "selection_reasons": [],
                "nearest_proxy_transition_step": transition_step,
                "pixel_proxy_battle_like": proxy_by_step.get(frame_step),
                "manual_visual_anchor": review_label,
            })
            if reason not in item["selection_reasons"]:
                item["selection_reasons"].append(reason)
            if review_label:
                item["manual_visual_anchor"] = review_label

        transitions = session.get("visual_transitions", [])
        for transition in transitions:
            at = int(transition["step"])
            before = [step for step in steps if step <= at]
            after = [step for step in steps if step >= at]
            for reason, candidates in (("closest frame at/before proxy transition", before), ("closest frame at/after proxy transition", after)):
                if candidates:
                    selected_step = min(candidates, key=lambda value: abs(value - at))
                    path = next(path for frame_step, path in frames if frame_step == selected_step)
                    add(selected_step, path, reason, at)
        for anchor in anchors:
            if anchor["session_id"] != session_id:
                continue
            found = _nearest_frame(frames, int(anchor["step"]))
            if found:
                add(found[0], found[1], "manual visual reference anchor", None, anchor["label"])
    return sorted(selected.values(), key=lambda row: (row["session_id"], row["frame_step"]))


def analyze_frame(item: dict) -> dict:
    with Image.open(item["image_path"]) as source:
        rgb = np.asarray(source.convert("RGB"))
    features = visual_layout_features(rgb)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()[:16]
    return {
        "session_id": item["session_id"],
        "frame_step": item["frame_step"],
        "pixel_proxy_battle_like": item["pixel_proxy_battle_like"],
        "nearest_proxy_transition_step": item["nearest_proxy_transition_step"],
        "selection_reasons": "; ".join(item["selection_reasons"]),
        "manual_visual_anchor": item["manual_visual_anchor"] or "",
        "visual_layout_family": visual_layout_family(features),
        "screen_sha256_prefix": digest,
        **{key: round(value, 4) for key, value in features.items()},
        "image_path": item["image_path"],
    }


def _contact_sheets(rows: list[dict], evidence_dir: Path, per_sheet: int = 12) -> list[dict]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    tile_w, tile_h, caption_h, columns = 112, 105, 27, 3
    for sheet_index, offset in enumerate(range(0, len(rows), per_sheet), start=1):
        group = rows[offset:offset + per_sheet]
        lines = math.ceil(len(group) / columns)
        sheet = Image.new("RGB", (columns * tile_w, lines * (tile_h + caption_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(group):
            with Image.open(row["image_path"]) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((tile_w, tile_h))
            x, y = (index % columns) * tile_w, (index // columns) * (tile_h + caption_h)
            sheet.paste(thumb, (x + (tile_w - thumb.width) // 2, y))
            caption = f"{row['session_id']} step {row['frame_step']}"
            family = row["visual_layout_family"].replace("_layout_candidate", "")
            draw.text((x + 2, y + tile_h + 2), caption[:24], fill="black", font=ImageFont.load_default())
            draw.text((x + 2, y + tile_h + 15), family[:24], fill="black", font=ImageFont.load_default())
        sheet = sheet.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
        path = evidence_dir / f"transition_scenes_{sheet_index:02d}.png"
        sheet.save(path, format="PNG", optimize=True)
        manifest.append({"file": path.name, "frames": len(group), "bytes": path.stat().st_size})
    return manifest


def run_audit(capture_dir: Path, output_dir: Path, anchors: list[dict] | None = None) -> Path:
    capture_dir = Path(capture_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit package: {output_dir}")
    metrics_path = capture_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing capture metrics: {metrics_path}")
    capture_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if capture_metrics.get("status") != "complete":
        raise ValueError("Scene audit requires a completed capture")
    anchors = anchors or []
    output_dir.mkdir(parents=True)
    selected = select_review_frames(capture_dir, capture_metrics, anchors)
    rows = [analyze_frame(item) for item in selected]
    fields = [
        "session_id", "frame_step", "pixel_proxy_battle_like", "nearest_proxy_transition_step",
        "selection_reasons", "manual_visual_anchor", "visual_layout_family", "screen_sha256_prefix",
        "green_fraction", "dark_fraction", "bright_fraction",
        "right_party_panel_edge_brightness", "center_menu_region_brightness",
    ]
    with (output_dir / "frame_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    evidence = _contact_sheets(rows, output_dir / "evidence")
    family_counts = Counter(row["visual_layout_family"] for row in rows)
    dark_proxy_battles = [row for row in rows if row["visual_layout_family"] == "dark_menu_or_transition_layout_candidate" and row["pixel_proxy_battle_like"]]
    anchor_rows = [row for row in rows if row["manual_visual_anchor"]]
    metrics = {
        "run_id": capture_metrics["run_id"],
        "capture_sessions": len(capture_metrics.get("episodes", [])),
        "capture_samples": sum(int(item.get("steps_recorded", 0)) for item in capture_metrics.get("episodes", [])),
        "selected_review_frames": len(rows),
        "layout_family_counts": dict(sorted(family_counts.items())),
        "dark_menu_or_transition_proxy_battle_frames": [
            {"session_id": row["session_id"], "frame_step": row["frame_step"]}
            for row in dark_proxy_battles
        ],
        "manual_visual_anchors": [
            {"session_id": row["session_id"], "frame_step": row["frame_step"], "label": row["manual_visual_anchor"], "layout_family": row["visual_layout_family"]}
            for row in anchor_rows
        ],
        "uses_ram": False,
        "labels_are_ground_truth": False,
        "evidence_sheets": evidence,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# NES pixel-only scene audit — {capture_metrics['run_id']}",
        "",
        "This audit reads only saved screenshots and the pixel-proxy transition timestamps. It does not read RAM. Coarse layout families prioritize review and are not validated game-state labels.",
        "",
        f"- Source capture: {len(metrics['capture_sessions']) if isinstance(metrics['capture_sessions'], list) else metrics['capture_sessions']} sessions, {metrics['capture_samples']:,} samples.",
        f"- Selected {len(rows)} saved frames around pixel-proxy transitions plus {len(anchor_rows)} explicit visual anchors.",
        f"- Layout hypotheses among selected frames: {dict(sorted(family_counts.items()))}.",
        f"- Dark menu/transition layout candidates labeled battle-like by the old proxy: {len(dark_proxy_battles)}; see frame_features.csv for steps.",
        "",
        "## Independently reviewed visual anchors",
        "",
        "| Session | Step | Human visual label | Pixel-only layout cue |",
        "|---|---:|---|---|",
    ]
    for row in anchor_rows:
        report.append(f"| {row['session_id']} | {row['frame_step']} | {row['manual_visual_anchor']} | {row['visual_layout_family']} |")
    report.extend([
        "",
        "Two manually reviewed title screens (session 1 step 1680 and session 2 step 1360) were flagged battle-like by the old pixel proxy. The dark-menu cue intentionally groups title, menus, transitions, and possible game-over screens; it does not separate them. Likewise, a battle-panel layout does not prove the game is awaiting a player action rather than showing a result or transition.",
        "",
        "## Current limitations and decision",
        "",
        "No independent labels exist yet for battle victory, defeat, escape, or game-over. The capture does not expose validated HP, EXP, location, or progress signals. Review the small transition sheets and label a handful of true result screens before promoting any visual or RAM cue. Do not change PPO rewards or start training from these candidates.",
        "",
        "Only aggregate visual features, selected frame metadata, and downscaled contact sheets are included. Original frames and raw RAM remain in the local ignored capture.",
    ])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output_dir


def parse_anchor(value: str) -> dict:
    try:
        left, label = value.split("=", 1)
        session_id, raw_step = left.rsplit(":", 1)
        step = int(raw_step)
    except (ValueError, TypeError) as exc:
        raise ValueError("Anchor must be SESSION:STEP=LABEL") from exc
    if not session_id or step < 0 or not label.strip():
        raise ValueError("Anchor must include a session, nonnegative step, and label")
    return {"session_id": session_id, "step": step, "label": label.strip()}
