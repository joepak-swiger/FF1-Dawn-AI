from pathlib import Path
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from uuid import uuid4

import numpy as np
from PIL import Image
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ff1_dawn_ai.nes_env import FF1NesEnv


OUTPUT_ROOT = PROJECT_ROOT / "videos" / "nes_ram_verify_v1_0"

CANDIDATE_ADDRESSES = [
    0x0014,
    0x00D2, 0x00D3,
    0x03C1, 0x03C2, 0x03C3,
    0x03CD, 0x03CE, 0x03CF,
    0x03D1, 0x03D2, 0x03D3, 0x03D5, 0x03D9, 0x03DA, 0x03DB, 0x03DD, 0x03DE, 0x03DF,
    0x0270, 0x0271, 0x0272, 0x0273, 0x0274, 0x0275,
    0x00B5, 0x00D5, 0x00C5, 0x00D4, 0x00DC, 0x00DF,
]

LABEL_KEYS = {
    pygame.K_F1: "field_overworld",
    pygame.K_F2: "battle",
    pygame.K_F3: "menu_town",
    pygame.K_F4: "other",
}

CLEAR_LABEL_KEY = pygame.K_F6

ACTION_KEYS = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_z: 4,
    pygame.K_a: 4,
    pygame.K_x: 5,
    pygame.K_s: 5,
}

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]


def make_env():
    return FF1NesEnv(
        project_root=PROJECT_ROOT,
        max_steps=100000,
        action_freq=8,
        print_buttons=False,
    )


def read_ram(env):
    objects_to_try = [
        ("env", env),
        ("env.retro_env", getattr(env, "retro_env", None)),
        ("env.retro_env.unwrapped", getattr(getattr(env, "retro_env", None), "unwrapped", None)),
    ]

    for label, obj in objects_to_try:
        if obj is None:
            continue

        getter = getattr(obj, "get_ram", None)
        if callable(getter):
            ram = np.asarray(getter(), dtype=np.uint8).reshape(-1)
            if ram.size > 0:
                return ram, f"{label}.get_ram()"

        ram_attr = getattr(obj, "ram", None)
        if ram_attr is not None:
            ram = np.asarray(ram_attr, dtype=np.uint8).reshape(-1)
            if ram.size > 0:
                return ram, f"{label}.ram"

    raise RuntimeError("Could not read RAM from environment.")


def save_frame(rgb, step, label, reason, frame_dir):
    path = frame_dir / f"step_{step:05d}_{label}_{reason}.png"
    Image.fromarray(rgb).save(path)
    return path


def values_for_addresses(ram):
    row = {}
    for address in CANDIDATE_ADDRESSES:
        key = f"ram_{address:04X}"
        row[key] = int(ram[address]) if address < ram.size else None
    return row


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_value(values):
    if not values:
        return None, 0.0

    value, count = Counter(values).most_common(1)[0]
    return value, count / len(values)


def analyze(rows):
    """Summarize manually labeled samples without claiming address semantics.

    Epochs are started every time the user presses a label key. They provide a
    rough count of independently entered labeled intervals; they are not
    independent emulator resets or proof that a label was correct.
    """
    summary = []

    labels_seen = sorted(set(row["label"] for row in rows if row["label"] != "unlabeled"))

    for address in CANDIDATE_ADDRESSES:
        key = f"ram_{address:04X}"

        grouped = defaultdict(list)
        for row in rows:
            if row.get("sample_type", "action") != "action":
                continue
            label = row["label"]
            value = row.get(key)
            if label != "unlabeled" and value is not None:
                grouped[label].append(int(value))

        label_parts = {}
        all_values = []

        for label in labels_seen:
            values = grouped.get(label, [])
            all_values.extend(values)
            top, frac = top_value(values)
            epochs = {
                row.get("label_epoch")
                for row in rows
                if row.get("sample_type", "action") == "action"
                and row["label"] == label and row.get(key) is not None
            }
            label_parts[label] = {
                "count": len(values),
                "epochs": len(epochs),
                "top": top,
                "top_fraction": round(frac, 4),
                "unique_values": sorted(set(values))[:20],
            }

        field_top = label_parts.get("field_overworld", {}).get("top")
        battle_top = label_parts.get("battle", {}).get("top")
        field_frac = label_parts.get("field_overworld", {}).get("top_fraction", 0)
        battle_frac = label_parts.get("battle", {}).get("top_fraction", 0)

        has_field = label_parts.get("field_overworld", {}).get("count", 0) > 0
        has_battle = label_parts.get("battle", {}).get("count", 0) > 0

        differs = has_field and has_battle and field_top != battle_top
        separation_score = 0.0

        if has_field and has_battle:
            separation_score = field_frac + battle_frac + (1.0 if differs else 0.0)

        summary.append({
            "address_hex": f"0x{address:04X}",
            "address_dec": address,
            "separation_score": round(separation_score, 4),
            "field_top": field_top,
            "field_top_fraction": field_frac,
            "field_epochs": label_parts.get("field_overworld", {}).get("epochs", 0),
            "battle_top": battle_top,
            "battle_top_fraction": battle_frac,
            "battle_epochs": label_parts.get("battle", {}).get("epochs", 0),
            "field_battle_differ": differs,
            "unique_values_across_field_battle": len(set(all_values)),
            "labels_seen": json.dumps(label_parts),
        })

    summary.sort(key=lambda item: item["separation_score"], reverse=True)
    return summary


def draw_overlay(surface, font, step, label, action_name, ram):
    lines = [
        "FF1 RAM Verify v1.0",
        f"Step: {step}",
        f"Label: {label}",
        f"Last action: {action_name}",
        "",
        "Controls:",
        "Arrows = move",
        "Z/A = game A",
        "X/S = game B",
        "F1 field | F2 battle",
        "F3 menu/town | F4 other",
        "F5 screenshot | F6 clear label | Q quit",
        "Hold movement keys for repeated actions.",
        "",
        "Key RAM:",
    ]

    for address in CANDIDATE_ADDRESSES[:12]:
        value = int(ram[address]) if address < ram.size else -1
        lines.append(f"0x{address:04X}: {value}")

    y = 8
    for line in lines:
        text = font.render(line, True, (255, 255, 255))
        surface.blit(text, (8, y))
        y += 18


def main():
    parser = argparse.ArgumentParser(description="Interactively record manually labeled NES RAM states.")
    parser.add_argument(
        "--run-id",
        help="Optional unique folder name. Defaults to a timestamp plus random suffix; existing folders are never overwritten.",
    )
    args = parser.parse_args()
    run_id = args.run_id or (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        parser.error("--run-id must be 1-80 letters, digits, dots, underscores, or hyphens")
    out_dir = OUTPUT_ROOT / run_id
    frame_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=False)
    frame_dir.mkdir()

    created_at = datetime.now().isoformat(timespec="seconds")

    env = make_env()
    obs, info = env.reset()
    ram, ram_method = read_ram(env)

    pygame.init()
    pygame.display.set_caption("FF1 NES RAM Verify v1.0")

    scale = 3
    game_w = 240 * scale
    game_h = 224 * scale
    side_w = 360

    screen = pygame.display.set_mode((game_w + side_w, game_h))
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    current_label = "unlabeled"
    last_action_name = "RESET"
    step_count = 0
    label_epoch = 0
    rows = []
    snapshots = []
    held_action = None
    next_action_at = 0
    action_interval_ms = 100

    print()
    print("FF1 NES RAM verify v1.0")
    print("----------------------")
    print("A window should be open.")
    print("F1 field, F2 battle, F3 menu/town, F4 other, F5 screenshot, F6 clear label, Q quit.")
    print("Labels start a new epoch. Label every transition and clear the label while unsure.")
    print("Hold movement/action keys for repeated actions; a step is issued every 100 ms.")
    print()

    running = True

    while running:
        rgb = env.render()
        ram, _method = read_ram(env)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key in LABEL_KEYS:
                    current_label = LABEL_KEYS[event.key]
                    label_epoch += 1
                    path = save_frame(rgb, step_count, current_label, "label", frame_dir)
                    snapshots.append(str(path))
                    row = {
                        "step": step_count,
                        "sample_type": "label",
                        "label": current_label,
                        "label_epoch": label_epoch,
                        "action": "LABEL",
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                    }
                    row.update(values_for_addresses(ram))
                    rows.append(row)
                    print(f"Label epoch {label_epoch}: {current_label}. Saved {path}")

                elif event.key == CLEAR_LABEL_KEY:
                    current_label = "unlabeled"
                    label_epoch += 1
                    print(f"Label cleared; new epoch {label_epoch}.")

                elif event.key == pygame.K_F5:
                    path = save_frame(rgb, step_count, current_label, "snapshot", frame_dir)
                    snapshots.append(str(path))
                    print(f"Saved snapshot: {path}")


        pressed = pygame.key.get_pressed()
        held_action = next((ACTION_KEYS[key] for key in ACTION_KEYS if pressed[key]), None)
        now = pygame.time.get_ticks()
        if held_action is not None and now >= next_action_at:
            obs, reward, terminated, truncated, info = env.step(held_action)
            step_count += 1
            last_action_name = ACTION_NAMES[held_action]
            next_action_at = now + action_interval_ms

            ram_after, _method = read_ram(env)
            row = {
                "step": step_count,
                "sample_type": "action",
                "label": current_label,
                "label_epoch": label_epoch if current_label != "unlabeled" else None,
                "action": last_action_name,
                "reward": round(float(reward), 4),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            row.update(values_for_addresses(ram_after))
            rows.append(row)

            if step_count % 20 == 0:
                path = save_frame(env.render(), step_count, current_label, "auto", frame_dir)
                snapshots.append(str(path))

            if terminated or truncated:
                running = False

        rgb = env.render()
        ram, _method = read_ram(env)

        frame = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        frame = pygame.transform.scale(frame, (game_w, game_h))

        screen.fill((0, 0, 0))
        screen.blit(frame, (0, 0))
        draw_overlay(screen.subsurface((game_w, 0, side_w, game_h)), font, step_count, current_label, last_action_name, ram)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    env.close()

    events_path = out_dir / "manual_ram_events.csv"
    summary_path = out_dir / "candidate_address_label_summary.csv"
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "run_report.txt"

    write_csv(events_path, rows)
    summary = analyze(rows)
    write_csv(summary_path, summary)

    label_counts = Counter(row["label"] for row in rows if row["sample_type"] == "action")
    epoch_counts = Counter(
        row["label"] for row in rows
        if row["sample_type"] == "label" and row["label"] != "unlabeled"
    )

    metrics = {
        "created_at": created_at,
        "session_id": run_id,
        "ram_method": ram_method,
        "ram_size": int(ram.size),
        "steps_recorded": len(rows),
        "label_counts": dict(label_counts),
        "label_epoch_counts": dict(epoch_counts),
        "candidate_addresses": [f"0x{x:04X}" for x in CANDIDATE_ADDRESSES],
        "top_candidate_signals": summary[:20],
        "snapshots": snapshots,
        "files": {
            "manual_events_csv": str(events_path),
            "candidate_summary_csv": str(summary_path),
            "run_report": str(report_path),
            "frames_folder": str(frame_dir),
        },
    }

    metrics_path.write_text(json.dumps(metrics, indent=2))

    lines = [
        "FF1 NES RAM Verify v1.0",
        "=======================",
        "",
        f"Created: {created_at}",
        f"RAM method: {ram_method}",
        f"RAM size: {ram.size}",
        f"steps_recorded: {len(rows)}",
        f"label_counts: {dict(label_counts)}",
        "",
        "How to read this",
        "----------------",
        "Rows are grouped by explicit label epochs. Label every transition; clear",
        "the label while unsure. Epoch counts are not independent emulator runs.",
        "A differing modal value is only a candidate signal, not proof of meaning.",
        "",
        "Top labeled RAM candidate signals (not verified addresses)",
        "----------------------------------------------------------",
    ]

    for index, item in enumerate(summary[:25], start=1):
        lines.append(
            f"{index}. {item['address_hex']} "
            f"separation_score={item['separation_score']} "
            f"field={item['field_top']}({item['field_top_fraction']}) "
            f"battle={item['battle_top']}({item['battle_top_fraction']}) "
            f"epochs={item['field_epochs']}/{item['battle_epochs']} "
            f"differs={item['field_battle_differ']}"
        )

    lines.extend([
        "",
        "Files written",
        "-------------",
        f"metrics: {metrics_path}",
        f"manual_events_csv: {events_path}",
        f"candidate_summary_csv: {summary_path}",
        f"frames_folder: {frame_dir}",
        "",
        "Next likely step",
        "----------------",
        "Repeat labeled transitions across independent battles and emulator resets.",
        "Do not use a candidate in rewards until it agrees across repetitions and",
        "is checked against field, battle, menu/town, battle exit, and game-over states.",
    ])

    report_path.write_text("\n".join(lines) + "\n")

    print()
    print("v1.0 RAM verification complete.")
    print(f"Run report: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Candidate summary: {summary_path}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
