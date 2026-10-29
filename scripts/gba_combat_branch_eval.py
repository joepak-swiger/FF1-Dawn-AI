#!/usr/bin/env python3
"""Replay a captured GBA battle-menu state for bounded policy comparison.

This is a live Stable-Retro branch experiment, not an ordinary fresh-field
episode. It keeps the supplied native state unchanged and only saves expert
labels if the exact reviewed EXP victory banner appears.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_autoplay import (  # noqa: E402
    BattleCommandDetector, BattleConfirmationGate, BattleExpResultDetector,
    BattleTargetDetector,
)
from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402
from ff1_dawn_ai.gba_combat_learning import (  # noqa: E402
    CombatDecisionRecorder, CombatLinearPolicy,
)
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _new_output(root: Path) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        output = root / run_id
        try:
            output.mkdir()
            return run_id, output
        except FileExistsError:
            time.sleep(0.001)


def _save_contact_sheet(frames: list, labels: list[str], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not frames:
        return
    count = min(8, len(frames))
    indices = sorted({round(i * (len(frames) - 1) / max(1, count - 1)) for i in range(count)})
    width, height = 240, 180
    sheet = Image.new("RGB", (width * 2, height * ((len(indices) + 1) // 2)), "#111827")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for n, index in enumerate(indices):
        x, y = (n % 2) * width, (n // 2) * height
        sheet.paste(frames[index].resize((width, 160), Image.Resampling.NEAREST), (x, y + 18))
        draw.text((x + 3, y + 2), labels[index][:36], fill="white", font=font)
    sheet.save(path, optimize=True)


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    from PIL import Image

    if not args.state.is_file():
        raise FileNotFoundError(args.state)
    if args.policy == "learned" and (not args.model or not args.model.is_file()):
        raise ValueError("learned policy requires a local --model")
    if args.record_expert and args.policy != "scripted":
        raise ValueError("expert labels can only be collected from the scripted expert controller")

    attack_ref = ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/attack_command_positive_left.png"
    target_ref = ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/goblin_target_positive_left.png"
    magic_ref = ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/magic_menu_negative_left.png"
    exp_ref = ROOT / "docs/experiments/gba_autoplay/battle_win_reference/earned_6_exp_banner.png"
    detector = BattleCommandDetector(str(attack_ref), threshold=0.95)
    attack_gate = BattleConfirmationGate(detector, consecutive_frames=2)
    target_detector = BattleTargetDetector(str(target_ref), alternative_references=(
        str(ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/goblin_target_positive_four.png"),
    ), threshold=0.95)
    target_gate = BattleConfirmationGate(target_detector, consecutive_frames=2)
    magic_detector = BattleCommandDetector(str(magic_ref), threshold=0.95)
    magic_gate = BattleConfirmationGate(magic_detector, consecutive_frames=2)
    exp_detector = BattleExpResultDetector(str(exp_ref), alternative_references={
        4: str(ROOT / "docs/experiments/gba_autoplay/battle_win_reference/earned_4_exp_banner.png")},
        threshold=0.985)
    exp_gate = BattleConfirmationGate(exp_detector, consecutive_frames=2)

    state_bytes = args.state.read_bytes()
    parent_id = _sha(state_bytes)
    run_id, output = _new_output(ROOT / "videos" / "gba_combat_branch_eval")
    model = CombatLinearPolicy.load(args.model) if args.policy == "learned" else None
    model_sha = _sha(args.model.read_bytes()) if model is not None else None
    rng = random.Random(args.seed)
    recorder = CombatDecisionRecorder(output / "expert_decisions") if args.record_expert else None
    pending = []
    trace = []
    events = []
    frames = []
    frame_labels = []
    action_counts = {name: 0 for name in GBA_ACTIONS}
    meaningful_sources = {name: 0 for name in ("learned", "scripted_expert", "random", "safety_supervisor")}
    choice_attempts = {name: 0 for name in ("learned", "scripted_expert", "random")}
    rejected_choices = []
    decisions = 0
    exp_confirmed = False
    exp_amount = None
    stop_reason = "bounded_actor_limit"
    status = "running"

    def current_frame(env):
        frame = env.render().copy()
        return frame

    def overlay(frame, label, stage, action):
        from PIL import ImageDraw, ImageFont
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 19), fill=(0, 0, 0))
        draw.text((3, 3), f"{label.upper()} | {stage} | {action}",
                  fill=(255, 235, 120), font=ImageFont.load_default())
        return image

    env = FF1GbaEnv(discover_checkpoint(), project_root=ROOT,
                    max_steps=args.max_actors * 220 + args.resolution_frames + 16, action_frames=1,
                    render_mode="rgb_array", restore_count=2)
    started = time.perf_counter()
    try:
        env.reset(seed=args.seed)
        env.retro_env.em.set_state(state_bytes)
        for _ in range(2):
            env.step(GBA_ACTIONS.index("NOOP"))
        frame = current_frame(env)

        def inspect_gates(image):
            is_attack = attack_gate.update(image)
            is_target = target_gate.update(image)
            is_magic = magic_gate.update(image)
            exp_seen = exp_gate.update(image)
            return is_attack, is_target, is_magic, exp_seen

        def record_frame(stage: str, action: str) -> None:
            image = overlay(current_frame(env), args.policy, stage, action)
            if (decisions == 0 or decisions % args.capture_every == 0 or action != "NOOP"
                    or stage in {"magic_menu", "attack_command", "target_confirm", "victory_result"}):
                frames.append(image)
                frame_labels.append(f"{decisions}: {stage} {action}")

        def decision(stage: str, expert: str) -> tuple[str, str, float | None]:
            image = current_frame(env)
            if recorder is not None:
                pending.append({"frame": image.copy(), "stage": stage,
                                "action": expert, "group_id": parent_id,
                                "encounter_id": f"{run_id}:{parent_id[:12]}"})
            if args.policy == "scripted":
                action, source, confidence = expert, "scripted_expert", None
            elif args.policy == "random":
                candidates = ("A", "B")
                action, source, confidence = rng.choice(candidates), "random", None
            else:
                action, confidence = model.predict(image, stage)
                source = "learned"
                events.append({"event": "learned_action", "decision": decisions,
                               "stage": stage, "action": action,
                               "confidence": round(confidence, 6), "model_sha256": model_sha})
            choice_attempts[source] += 1
            events.append({"event": "policy_choice_attempted", "decision": decisions,
                           "stage": stage, "action": action, "source": source})
            return action, source, confidence

        def reject_choice(stage: str, action: str, source: str, expected: str) -> None:
            row = {"stage": stage, "action": action, "source": source,
                   "expected_action": expected}
            rejected_choices.append(row)
            events.append({"event": "policy_choice_rejected_by_safety_gate", **row})

        def step(control: str, phase: str, source: str, confidence=None):
            nonlocal decisions, exp_confirmed, exp_amount
            if control not in GBA_ACTIONS:
                raise ValueError(f"unsupported control {control!r}")
            env.step(GBA_ACTIONS.index(control))
            action_counts[control] += 1
            if control != "NOOP":
                meaningful_sources[source] += 1
            decisions += 1
            frame = current_frame(env)
            attack_ok, target_ok, magic_ok, exp_ok = inspect_gates(frame)
            trace.append({"decision": decisions, "action": control,
                          "phase": phase, "source": source,
                          "model_confidence": confidence if confidence is not None else "",
                          "model_sha256": model_sha if source == "learned" else "",
                          "attack_anchor": round(detector.score(frame), 5),
                          "target_anchor": round(target_detector.score(frame), 5),
                          "magic_anchor": round(magic_detector.score(frame), 5),
                          "exp_anchor": round(exp_detector.score(frame), 5)})
            if exp_ok and not exp_confirmed:
                exp_confirmed = True
                exp_amount = exp_detector.recognized_amount(frame)
                Image.fromarray(frame).save(output / f"victory_earned_{exp_amount or 'unknown'}_exp.png")
                events.append({"event": "reviewed_exp_banner_confirmed", "amount": exp_amount,
                               "decision": decisions})
            record_frame(phase, control)
            return frame, attack_ok, target_ok, magic_ok

        # The two framebuffer-refresh steps are also the two distinct
        # observations required to confirm the loaded menu. Do not advance a
        # second pair of frames, which would alter the captured decision state.
        for _ in range(2):
            frame, *_ = step("NOOP", "state_sync", "safety_supervisor")
        initial = "magic_menu" if magic_gate.confirmed else (
            "attack_command" if attack_gate.confirmed else
            "target_confirm" if target_gate.confirmed else "unknown")
        events.append({"event": "initial_state_verified", "state": initial,
                       "magic_anchor": round(magic_detector.score(frame), 5),
                       "attack_anchor": round(detector.score(frame), 5),
                       "target_anchor": round(target_detector.score(frame), 5)})
        if initial == "unknown":
            status, stop_reason = "unrecognized_start_state", "input_state_matches_no_reviewed_battle_panel"
        else:
            # Complete the currently displayed actor, then handle at most the
            # remaining configured actor panels. Every transition is gated.
            actor_count = 0
            current_state = initial
            while actor_count < args.max_actors and not exp_confirmed:
                if current_state == "magic_menu":
                    action, source, confidence = decision("menu_other", "B")
                    if action != "B":
                        reject_choice("menu_other", action, source, "B")
                        status, stop_reason = "safe_stop", "policy_did_not_choose_verified_magic_back"
                        break
                    magic_gate.reset()
                    attack_gate.reset()
                    target_gate.reset()
                    frame, _, _, _ = step(action, "magic_menu_back", source, confidence)
                    current_state = "unknown"
                    for _ in range(30):
                        frame, attack_ok, target_ok, magic_ok = step("NOOP", "wait_attack_command", "safety_supervisor")
                        # A persistent Magic match is expected while the
                        # dismissal animation settles; keep waiting for the
                        # independently confirmed Attack panel.
                        if attack_ok or target_ok:
                            current_state = ("attack_command" if attack_gate.confirmed else
                                             "target_confirm" if target_gate.confirmed else "unknown")
                            break
                    Image.fromarray(frame).save(output / "after_magic_back_wait.png")
                    if current_state not in ("attack_command", "target_confirm"):
                        status, stop_reason = "safe_stop", "magic_back_did_not_restore_attack_command"
                        break
                    events.append({"event": "magic_back_reviewed_panel_confirmed",
                                   "panel": current_state, "decision": decisions})

                if current_state == "attack_command":
                    action, source, confidence = decision("attack_command", "A")
                    if action != "A":
                        reject_choice("attack_command", action, source, "A")
                        status, stop_reason = "safe_stop", "policy_action_not_verified_for_attack_panel"
                        break
                    attack_gate.reset()
                    target_gate.reset()
                    frame, _, _, _ = step(action, "attack_select", source, confidence)
                    current_state = "unknown"
                    for _ in range(30):
                        frame, attack_ok, target_ok, magic_ok = step("NOOP", "wait_target_panel", "safety_supervisor")
                        # The Attack cursor remains visible briefly after A;
                        # don't mistake its transition latency for a failed
                        # command. Wait for the target or the reviewed Magic
                        # submenu before deciding what happened.
                        if target_ok or magic_ok:
                            current_state = "target_confirm" if target_gate.confirmed else "magic_menu"
                            break
                    Image.fromarray(frame).save(output / "after_attack_select_wait.png")
                    if current_state != "target_confirm":
                        status, stop_reason = "safe_stop", "attack_did_not_reach_reviewed_target_panel"
                        break

                if current_state == "target_confirm":
                    action, source, confidence = decision("target_confirm", "A")
                    if action != "A":
                        reject_choice("target_confirm", action, source, "A")
                        status, stop_reason = "safe_stop", "policy_action_not_verified_for_target_panel"
                        break
                    target_gate.reset()
                    attack_gate.reset()
                    frame, _, _, _ = step(action, "target_confirm", source, confidence)
                    actor_count += 1
                    current_state = "unknown"
                    for _ in range(180):
                        frame, attack_ok, target_ok, magic_ok = step("NOOP", "wait_next_actor", "safety_supervisor")
                        if attack_ok or magic_ok or exp_confirmed:
                            current_state = "attack_command" if attack_gate.confirmed else "magic_menu"
                            break
                    if exp_confirmed:
                        break
                    if current_state == "unknown":
                        # Some encounters need another command round after a
                        # valid target action. Wait for either a new reviewed
                        # command panel or the independent EXP result.
                        for _ in range(args.resolution_frames):
                            frame, attack_ok, _target_ok, _magic_ok = step(
                                "NOOP", "battle_resolution_wait", "safety_supervisor")
                            if exp_confirmed:
                                break
                            if attack_ok:
                                current_state = "attack_command"
                                break
                        Image.fromarray(frame).save(output / "after_battle_resolution_wait.png")
                        if not exp_confirmed:
                            if current_state == "attack_command":
                                continue
                            status, stop_reason = "safe_stop", "no_next_command_or_exp_result_within_budget"
                        break
            if not exp_confirmed and status == "running":
                for _ in range(args.resolution_frames):
                    frame, *_ = step("NOOP", "battle_resolution_wait", "safety_supervisor")
                    if exp_confirmed:
                        break
                Image.fromarray(frame).save(output / "after_battle_resolution_wait.png")
            if exp_confirmed:
                status, stop_reason = "victory_confirmed", "exact_reviewed_exp_banner_twice_matched"
                if recorder is not None:
                    for sample in pending:
                        recorder.record(**sample)
                events.append({"event": "expert_examples_committed_after_victory",
                               "count": recorder.count if recorder else 0,
                               "parent_group_sha256": parent_id})
            elif status == "running":
                status, stop_reason = "bounded_stop", "actor_or_frame_budget_reached_without_result"
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    total_meaningful = sum(meaningful_sources.values())
    percentages = {key: round(value * 100.0 / total_meaningful, 2) if total_meaningful else 0.0
                   for key, value in meaningful_sources.items()}
    metrics = {
        "run_id": run_id, "status": status, "stop_reason": stop_reason,
        "policy": args.policy, "policy_is_learned": args.policy == "learned",
        "model_sha256": model_sha, "seed": args.seed,
        "source_state_sha256": parent_id,
        "starting_panel": initial if "initial" in locals() else "unknown",
        "meaningful_decisions_by_source": meaningful_sources,
        "meaningful_decision_source_percentages": percentages,
        "policy_decision_attempts": choice_attempts,
        "policy_choices_rejected_by_safety_gate": rejected_choices,
        "action_counts": action_counts, "decisions": decisions,
        "actor_command_sequences": actor_count if "actor_count" in locals() else 0,
        "exp_banner_confirmed": exp_confirmed, "recognized_exp_amount": exp_amount,
        "expert_labels_collected": recorder.count if recorder else 0,
        "training_timesteps": 0, "elapsed_seconds": round(elapsed, 3),
        "events": events,
        "interpretation": "Mid-battle state-branch replay; results do not establish field navigation or novel-species generalization.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "actions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("decision", "action", "phase", "source", "model_confidence", "model_sha256",
                  "attack_anchor", "target_anchor", "magic_anchor", "exp_anchor")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trace)
    label = "\n".join([
        "# GBA combat branch evaluation", f"Run ID: {run_id}",
        f"Policy: {args.policy}; model SHA-256: {model_sha or 'none'}",
        f"Starting panel: {metrics['starting_panel']}; confirmed EXP: {exp_confirmed} ({exp_amount})",
        f"Meaningful decisions by source: {meaningful_sources} ({percentages}%)",
        f"Actor command sequences: {metrics['actor_command_sequences']}; decisions: {decisions}; runtime: {elapsed:.3f}s",
        f"Stop reason: {stop_reason}", metrics["interpretation"], "",
    ])
    (output / "run_report.txt").write_text(label, encoding="utf-8")
    from PIL import Image
    gif_frames = frames
    if gif_frames:
        gif_frames[0].save(output / "combat_branch.gif", save_all=True,
                           append_images=gif_frames[1:], duration=90, loop=0,
                           optimize=True, disposal=2)
        _save_contact_sheet(frames, frame_labels, output / "contact_sheet.png")
    if args.publish:
        public = ROOT / "docs/experiments/gba_combat_learning" / run_id
        public.mkdir(parents=True, exist_ok=False)
        safe_metrics = {key: value for key, value in metrics.items()
                        if key not in {"events", "source_state_sha256"}}
        (public / "metrics.json").write_text(json.dumps(safe_metrics, indent=2) + "\n", encoding="utf-8")
        (public / "REPORT.md").write_text(label, encoding="utf-8")
        if (output / "contact_sheet.png").is_file():
            (public / "contact_sheet.png").write_bytes((output / "contact_sheet.png").read_bytes())
    print(f"Combat branch output: {output}")
    print(f"Recording: {output / 'combat_branch.gif'}")
    print(f"Status: {status}; victory={exp_confirmed}; decisions={decisions}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path, help="local native mid-battle Stable-Retro state")
    parser.add_argument("--policy", choices=("scripted", "random", "learned"), default="scripted")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--record-expert", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-actors", type=int, choices=range(1, 17), default=12,
                        help="Maximum bounded Attack/target sequences, including later battle rounds")
    parser.add_argument("--resolution-frames", type=int, default=1200)
    parser.add_argument("--capture-every", type=int, default=8)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"COMBAT BRANCH EVALUATION FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
