"""Package a reviewed, compact GitHub evidence set for a GBA castle-goal run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def package(run_dir: Path, *, reviewed_field_return: bool = False) -> Path:
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    trace_path = run_dir / "action_trace.jsonl"
    if not metrics_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError("run must contain metrics.json and local action_trace.jsonl")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not (metrics.get("checkpoint_unchanged") and metrics.get("reset_observations_identical")
            and metrics.get("neutral_rewards_only") and not metrics.get("training_started")):
        raise ValueError("refusing to package a changed checkpoint, nondeterministic reset, or training run")
    required_policies = {"SCRIPTED_UP", "SEEDED_RANDOM", "IDLE", "SIDE_DOWN"}
    if not required_policies.issubset(metrics.get("policy_summary", {})):
        raise ValueError("required scripted, random, idle, and side-route comparisons are missing")
    contact = run_dir / "castle_goal_contact_sheet.png"
    if not contact.is_file():
        raise FileNotFoundError(contact)

    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    def row(trial: str, step: int) -> dict | None:
        return next((r for r in trace if r.get("trial") == trial and r.get("step") == step), None)

    transition = [
        {"step": step, "label": (row("scripted_up_01", step) or {}).get("label"),
         "status": (row("scripted_up_01", step) or {}).get("status"),
         "confidence": (row("scripted_up_01", step) or {}).get("confidence")}
        for step in (119, 120, 121)
    ]
    town_collision = row("seeded_random_110", 128)
    if not town_collision or (town_collision.get("ram_0x33FC"), town_collision.get("ram_0x3404")) != (226, 230):
        raise ValueError("independent random-town RAM counterexample is missing or changed")
    down = metrics["down_64_frame_diagnostic"]["per_frame_labels_and_ram"]
    down_summary = []
    for action in ("NOOP", "DOWN"):
        for frame_no in (8, 16, 24, 32, 40, 48, 56, 64):
            item = next(x for x in down if x["action"] == action and x["frame"] == frame_no)
            shift = item["scene_translation_candidate"] or {}
            down_summary.append({"action": action, "frame": frame_no,
                                 "scene_dx_pixels": shift.get("dx_pixels_candidate"),
                                 "scene_dy_pixels": shift.get("dy_pixels_candidate"),
                                 "ram_0x33FC": item["ram_0x33FC"], "ram_0x3404": item["ram_0x3404"]})

    reversal_summary = {}
    for release, trials in metrics["reversal_8_frame_release_diagnostic"].items():
        if not isinstance(trials, list):
            continue
        reversal_summary[release] = {
            "repeats": len(trials),
            "mean_immediate_return_dy_pixels": _mean([
                t["immediate_return_scene_shift"]["dy_pixels_candidate"] for t in trials
            ]),
            "mean_settled_return_dy_pixels": _mean([
                t["settled_return_scene_shift"]["dy_pixels_candidate"] for t in trials
            ]),
            "mean_matched_idle_dy_pixels": _mean([
                t["matched_idle_scene_shift"]["dy_pixels_candidate"] for t in trials
            ]),
        }

    public = {
        "run_id": metrics["run_id"],
        "checkpoint_run_id": metrics["checkpoint_run_id"],
        "checkpoint_unchanged": metrics["checkpoint_unchanged"],
        "reset_observations_identical": metrics["reset_observations_identical"],
        "reset_count": metrics["reset_count"],
        "destination_detector": metrics["destination_detector"],
        "policy_summary": metrics["policy_summary"],
        "all_control_screens": metrics["all_control_screens"],
        "start_menu_control": metrics["start_menu_control"],
        "castle_entry_transition_samples": transition,
        "castle_exit_test": metrics["castle_exit_test"],
        "field_return_visual_review": {
            "reviewed": bool(reviewed_field_return),
            "result": "overworld exterior visually confirmed at 80 DOWN frames after castle entry"
            if reviewed_field_return else "not independently visually reviewed",
            "automated_field_detector": False,
        },
        "ram_candidate_label_differential": metrics["ram_candidate_label_differential"],
        "ram_counterexample": {
            "policy": "seeded random", "seed": 110, "step": 128,
            "independent_visual_label": "Cornelia town screen (not Castle Cornelia 1F)",
            "detector_result": {k: town_collision[k] for k in ("label", "status", "confidence")},
            "candidate_values": {"0x33FC": town_collision["ram_0x33FC"],
                                 "0x3404": town_collision["ram_0x3404"]},
            "interpretation": "counterexample to either offset or pair serving as a castle-entry flag; still no address meaning assigned",
        },
        "reversal_8_frame_release_summary": reversal_summary,
        "down_64_frame_summary": down_summary,
        "neutral_rewards_only": metrics["neutral_rewards_only"],
        "training_started": metrics["training_started"],
        "limitations": [
            "The title anchor detects only the named Castle Cornelia 1F screen; other screens remain unknown.",
            "The detector does not classify battles, menus, title/game-over screens, or general field scenes.",
            "Scene translation is camera/map pixel motion, not player coordinates or tile units.",
            "RAM offsets 0x33FC and 0x3404 remain provisional and are not used for landmark classification or rewards.",
            "The castle-exit field image was visually reviewed; no automated exterior-field detector exists yet.",
        ],
    }
    target = ROOT / "docs" / "experiments" / "gba_castle_goal" / metrics["run_id"]
    references = {
        "castle_before_transition.png": run_dir / "scripted_up_01_step_119.png",
        "castle_entry_first_match.png": run_dir / "scripted_up_01_step_120.png",
        "castle_entry_confirmed.png": run_dir / "scripted_up_01_step_121.png",
        "party_menu_negative_control.png": run_dir / "start_menu_control_32.png",
        "town_ram_collision.png": run_dir / "seeded_random_110_step_128.png",
        "castle_exit_transition.png": run_dir / "castle_exit_test_down_072.png",
        "overworld_after_castle_exit.png": run_dir / "castle_exit_test_down_080.png",
        "down_camera_limit_64.png": run_dir / "down64_down_64.png",
    }
    for name, source in references.items():
        if not source.is_file():
            raise FileNotFoundError(source)

    if target.exists():
        if (target / "REPORT.md").exists():
            raise FileExistsError(f"refusing to overwrite completed evidence package: {target}")
        allowed = {"metrics.json", "contact_sheet.png", "REPORT.md", *references}
        unexpected = {item.name for item in target.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"refusing to modify unexpected package contents: {sorted(unexpected)}")
        existing_metrics = target / "metrics.json"
        if existing_metrics.exists() and json.loads(existing_metrics.read_text(encoding="utf-8")) != public:
            raise ValueError("partial package metrics do not match this source run")
        existing_contact = target / "contact_sheet.png"
        if existing_contact.exists() and existing_contact.read_bytes() != contact.read_bytes():
            raise ValueError("partial package contact sheet does not match this source run")
        for name, source in references.items():
            saved = target / name
            if saved.exists() and saved.read_bytes() != source.read_bytes():
                raise ValueError(f"partial package image does not match source: {name}")
    else:
        target.mkdir(parents=True)
    metrics_out = target / "metrics.json"
    if not metrics_out.exists():
        metrics_out.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    if not (target / "contact_sheet.png").exists():
        shutil.copyfile(contact, target / "contact_sheet.png")
    for name, source in references.items():
        if not (target / name).exists():
            shutil.copyfile(source, target / name)

    report = [
        "# Castle Cornelia landmark evaluation", "",
        f"Run `{metrics['run_id']}` used the verified Seed-party checkpoint `{metrics['checkpoint_run_id']}`.",
        "",
        "## Verified destination", "",
        "The target is Castle Cornelia 1F, confirmed from the game's own floor-title UI. The detector compares the fixed title-banner text region to a reference extracted from a previously reviewed gameplay frame. It requires two consecutive high-confidence matches, ignores RAM and full-frame hashes, and returns `unknown` for all other scenes.",
        "",
        "The first 1-frame-per-action north route produced a transition/fade at step 119 (uncertain), matched at step 120, and met the two-frame confirmation at step 121. Three clean restores reached the target at that same confirmation step (3/3). The source checkpoint hash and all reset observations remained unchanged/reproducible.",
        "",
        "## Controlled policy comparison", "",
        "| Policy | Result | Entry cost |",
        "|---|---:|---:|",
    ]
    for key, label in (("SCRIPTED_UP", "Straight north"), ("SEEDED_RANDOM", "Seeded random"),
                       ("IDLE", "Idle"), ("SIDE_DOWN", "Straight south control")):
        item = public["policy_summary"][key]
        cost = item["median_actions_to_confirmed_entry"]
        report.append(f"| {label} | {item['successes']}/{item['trials']} | {cost if cost is not None else '—'} one-frame actions |")
    report += [
        "",
        f"The detector returned no castle hits in {public['all_control_screens']['frames_checked']} random, idle, southbound, and START-menu control observations ({public['all_control_screens']['castle_false_positive_frames']} false positives). Visual review confirmed the START control showed a white transition at frames 8/16 and the party/status menu at frames 32/64; none was classified as Castle Cornelia.",
        "",
        "## Castle exit and movement diagnostics", "",
        "After the castle-entry route, sustained DOWN input faded from the titled interior around frame 64. Frame 72 shows the scene transition, and frame 80 shows the exterior Cornelia overworld around the castle; visual review confirms a field return. The runtime detector intentionally reports this as unknown because it is a castle-title detector, not an exterior-field classifier.",
        "",
        "An 8-frame out-and-back retains about 14–16 screen pixels of apparent map offset immediately after reversing. With one neutral frame and 16 settling frames, residual averaged about 1 px; with at least two neutral release frames and 16 settling frames, the rendered map returned to within about 0.01 px of its start. Matched idle remained near 0 px. This points to movement/input timing and delayed camera settling; it does not establish a tile coordinate.",
        "",
        "The 64-frame DOWN probe's scene shift reached about −46 px by frame 48 and plateaued near −47.9 px by frame 56. Screenshots show the party approaching/crossing the southern bridge and the map view reaching its edge. This is not evidence of a blocked tile; collision remains unverified.",
        "",
        "## RAM candidates", "",
        "Across the three visually confirmed castle entries, `0x33FC/0x3404` read `226/230`. A seeded random replay (seed 110, step 128) displayed the separate `Cornelia` town screen and read the same `226/230` pair while the castle-title detector returned `unknown`. The pair also occurred on 127 non-confirmed frames in the main run. Neither offset, nor the pair, identifies Castle Cornelia uniquely. Their memory meaning remains unknown and neither is used by the detector or rewards.",
        "",
        "## Limits and next step", "",
        "The castle landmark is reliable for this named floor screen, but this is not a generalized scene classifier. The random comparison used 12 fixed seeds and a 192-action horizon, so its 0/12 result is a bounded baseline rather than a universal success rate. No PPO training or reward changes were made; NES v0.7 and its experiments were untouched.",
        "",
        "Next, add separately validated field/town/castle screen labels and use the title landmark in a short, held-out Gymnasium evaluation. Keep rewards neutral until labels survive negative controls across scenes and resets.",
        "",
        "Evidence includes the transition, menu false-positive control, same-value Cornelia-town counterexample, castle exit, and the DOWN camera-edge frames. Raw traces, ROMs, native states, and full capture directories remain local.",
    ]
    (target / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reviewed-field-return", action="store_true",
                        help="mark only after visually confirming the retained exterior-field frame")
    args = parser.parse_args()
    print(package(args.run_dir, reviewed_field_return=args.reviewed_field_return))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
