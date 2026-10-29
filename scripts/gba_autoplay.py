#!/usr/bin/env python3
"""Run a bounded, recorded GBA autoplay baseline from the official party state.

The scripted baseline repeats the measured UP-to-Castle / DOWN-to-exterior
excursion. A seeded-random mode is provided for encounter scouting. Neither is
represented as learned. Unknown screens and battle results remain unknown.
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
    BattleCommandDetector,
    BattleConfirmationGate,
    BattleExpResultDetector,
    BattleGilResultDetector,
    BattleTargetDetector,
    CorneliaPostBattleMapDetector,
    ScriptedCastleExcursion,
    SeededRandomExplorer,
    conservative_battle_outcome,
)


def discover_checkpoint() -> Path:
    """Reuse the existing official-roster/readiness validation."""
    from scripts.gba_gym_smoke import discover_checkpoint

    return discover_checkpoint()


def create_output(parent: Path) -> tuple[str, Path]:
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        out = parent / run_id
        try:
            out.mkdir()
            return run_id, out
        except FileExistsError:
            time.sleep(0.001)


def _overlay(frame, *, policy: str, phase: str, step: int, action: str,
             scene: str, battle_count: int):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # Keep the top status stripe visible: Dawn of Souls displays explicit
    # EXP-award text there, which is the current independent victory signal.
    draw.rectangle((0, 18, image.width, 53), fill=(0, 0, 0))
    draw.text((4, 20), f"POLICY: {policy.upper()} | {phase.upper()} | STEP {step}",
              fill=(255, 255, 255), font=font)
    draw.text((4, 36), f"ACTION: {action} | SCENE: {scene} | BATTLES: {battle_count}*",
              fill=(255, 230, 120), font=font)
    return image


def _write_gif(frames, path: Path, *, duration_ms: int = 60) -> None:
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms,
                    loop=0, optimize=True, disposal=2)


def _contact_sheet(frames, path: Path, labels: list[str]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not frames:
        return
    count = min(8, len(frames))
    indices = sorted({round(i * (len(frames) - 1) / max(1, count - 1)) for i in range(count)})
    thumb_w, thumb_h = 240, 160
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + 22) * ((len(indices) + 1) // 2)), "#111827")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for n, index in enumerate(indices):
        x = (n % 2) * thumb_w
        y = (n // 2) * (thumb_h + 22)
        sheet.paste(frames[index].resize((thumb_w, thumb_h), Image.Resampling.NEAREST), (x, y + 20))
        draw.text((x + 3, y + 3), labels[index][:38], fill="white", font=font)
    sheet.save(path, optimize=True)


def _public_package(out: Path, metrics: dict, report: str) -> Path:
    """Write the compact reviewer package; never copy native state or raw RAM."""
    public_dir = ROOT / "docs" / "experiments" / "gba_autoplay" / out.name
    public_dir.mkdir(parents=True, exist_ok=False)
    safe = {
        "run_id": metrics["run_id"],
        "status": metrics["status"],
        "policy": metrics["policy"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "decisions": metrics["decisions"],
        "emulator_frames": metrics["emulator_frames"],
        "confirmed_scenes": metrics["confirmed_scenes"],
        "castle_reached": metrics["castle_reached"],
        "field_return_confirmed": metrics["field_return_confirmed"],
        "battle_command_screen_confirmed": metrics["battle_command_screen_confirmed"],
        "battle_command_actions": metrics["battle_commands"],
        "battle_target_confirmed": metrics["battle_target_confirmed"],
        "next_party_command_confirmed": metrics["next_party_command_confirmed"],
        "post_target_wait_frames": metrics["post_target_wait_frames"],
        "battle_A_presses": metrics["battle_A_presses"],
        "battle_command_panel_returned": metrics["battle_command_panel_returned"],
        "field_return_after_battle": metrics["field_return_after_battle"],
        "battle_resolution_wait_frames": metrics["battle_resolution_wait_frames"],
        "battle_detection": metrics["battle_detection"],
        "battle_outcomes": metrics["battle_outcomes"],
        "specific_earned_6_exp_banner_confirmed": metrics["specific_earned_6_exp_banner_confirmed"],
        "specific_obtained_24_gil_banner_confirmed": metrics["specific_obtained_24_gil_banner_confirmed"],
        "post_battle_map_view_confirmed": metrics["post_battle_map_view_confirmed"],
        "unknown_scene_observations": metrics["unknown_scene_observations"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "training_timesteps": 0,
        "interpretation": "scripted or seeded-random control; not a learned policy",
    }
    (public_dir / "metrics.json").write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")
    (public_dir / "REPORT.md").write_text(report, encoding="utf-8")
    for filename in ("contact_sheet.png", "battle_command_confirmed.png",
                     "battle_command_after_A.png", "battle_after_target_A.png",
                     "battle_next_command_confirmed.png", "battle_result_earned_6_exp.png",
                     "battle_result_obtained_24_gil.png", "battle_postbattle_map_view.png"):
        source = out / filename
        if source.is_file():
            (public_dir / filename).write_bytes(source.read_bytes())
    return public_dir


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    from PIL import Image

    from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS
    from ff1_dawn_ai.gba_landmark import SceneAnchorClassifier, SceneConfirmationGate
    from scripts.gba_castle_learning_smoke import FIELD_VARIANTS, REFERENCES

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else discover_checkpoint().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official GBA checkpoint not found: {checkpoint}")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    classifier = SceneAnchorClassifier(REFERENCES, field_reference_variants=FIELD_VARIANTS)
    battle_reference = ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/attack_command_positive_left.png"
    battle_detector = BattleCommandDetector(str(battle_reference), threshold=0.95)
    battle_gate = BattleConfirmationGate(battle_detector, consecutive_frames=2)
    target_reference = ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/goblin_target_positive_left.png"
    target_detector = BattleTargetDetector(str(target_reference), threshold=0.95)
    target_gate = BattleConfirmationGate(target_detector, consecutive_frames=2)
    exp_reference = ROOT / "docs/experiments/gba_autoplay/battle_win_reference/earned_6_exp_banner.png"
    exp_detector = BattleExpResultDetector(str(exp_reference), threshold=0.985)
    exp_gate = BattleConfirmationGate(exp_detector, consecutive_frames=2)
    gil_reference = ROOT / "docs/experiments/gba_autoplay/battle_win_reference/obtained_24_gil_banner.png"
    gil_detector = BattleGilResultDetector(str(gil_reference), threshold=0.985)
    gil_gate = BattleConfirmationGate(gil_detector, consecutive_frames=2)
    map_reference = ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_static_anchor.png"
    map_detector = CorneliaPostBattleMapDetector(str(map_reference), threshold=0.995)
    map_gate = BattleConfirmationGate(map_detector, consecutive_frames=2)
    run_id, out = create_output(ROOT / "videos" / "gba_autoplay")
    started = time.perf_counter()
    policy_label = "SCRIPTED" if args.policy == "scripted_baseline" else "SEEDED RANDOM"
    controller = ScriptedCastleExcursion(args.max_approach, args.max_return) if args.policy == "scripted_baseline" else None
    random_policy = SeededRandomExplorer(args.seed) if args.policy == "seeded_random" else None
    gate = SceneConfirmationGate(2)
    frames = []
    frame_labels = []
    actions = []
    events = []
    action_counts = {name: 0 for name in GBA_ACTIONS}
    unknown_count = 0
    battle_commands = 0
    battle_entered = False
    battle_target_confirmed = False
    battle_a_presses = 0
    target_match_scores = []
    next_command_confirmed = False
    post_target_wait_frames = 0
    battle_command_panel_returned = False
    field_return_after_battle = False
    battle_resolution_wait_frames = 0
    battle_victory_exp_confirmed = False
    battle_gil_award_confirmed = False
    post_battle_map_view_confirmed = False
    battle_confirmation_step = None
    battle_match_scores = []
    reset_hashes = []
    confirmed_scenes = []
    previous_confirmed = None
    reset_info = {}
    status = "running"
    stop_reason = "decision_limit"
    env = None
    initial_scene = None
    try:
        env = FF1GbaEnv(checkpoint, project_root=ROOT,
                        max_steps=args.decisions + args.resolution_wait_frames + 400,
                        action_frames=1, render_mode="rgb_array", restore_count=2)
        obs, reset_info = env.reset(seed=args.seed)
        reset_hashes.append(reset_info.get("observation_sha256"))
        second_obs, second_reset = env.reset(seed=args.seed)
        reset_hashes.append(second_reset.get("observation_sha256"))
        if reset_hashes[0] != reset_hashes[1]:
            raise RuntimeError("native checkpoint reset did not reproduce the same observation")
        reset_info = second_reset
        gate.reset()
        battle_gate.reset()
        # The initial anchor needs two distinct core observations. A neutral
        # release is also the known-safe setup action for this checkpoint.
        for _ in range(2):
            obs, _, terminated, truncated, info = env.step(GBA_ACTIONS.index("NOOP"))
            frame = env.render()
            result = classifier.classify(frame)
            update = gate.update(result)
            battle_gate.update(frame)
            if terminated or truncated:
                raise RuntimeError("emulator ended while stabilizing the checkpoint")
        initial_scene = gate.confirmed_scene
        if initial_scene:
            confirmed_scenes.append(initial_scene)
        if args.policy == "scripted_baseline" and initial_scene != "cornelia_exterior_field":
            raise RuntimeError(f"checkpoint reset scene was not independently confirmed: {initial_scene!r}")

        def capture(action: str, step_no: int, phase: str, frame, result, update):
            nonlocal unknown_count, previous_confirmed
            # The current candidate describes this frame. The gate's previous
            # confirmed scene is context only and must not relabel an active
            # battle as field through a long unknown interval.
            label = result.label if result.status == "candidate" else "unknown"
            if result.label == "unknown":
                unknown_count += 1
            if update.transitioned and gate.confirmed_scene not in confirmed_scenes:
                confirmed_scenes.append(gate.confirmed_scene)
            if update.transitioned:
                events.append({"step": step_no, "event": "confirmed_scene_transition",
                               "from": previous_confirmed, "to": gate.confirmed_scene,
                               "action": action})
                previous_confirmed = gate.confirmed_scene
            if update.first_castle_entry:
                events.append({"step": step_no, "event": "confirmed_exterior_to_castle_entry",
                               "action": action})
            # Battle substeps use fractional trace IDs to keep rows ordered;
            # sample by actual action count so combat and result frames are
            # not silently skipped by modulo on those fractional IDs.
            if (step_no == 0 or len(actions) % args.capture_every == 0
                    or update.transitioned or battle_victory_exp_confirmed):
                frames.append(_overlay(frame, policy=policy_label, phase=phase, step=step_no,
                                       action=action, scene=label, battle_count=int(battle_entered)))
                frame_labels.append(f"{step_no}: {phase} / {action} / {label}")

        def battle_input(control: str, phase: str, marker: float):
            """Execute and trace one battle-control observation."""
            nonlocal battle_a_presses, battle_victory_exp_confirmed
            obs, reward, terminated, truncated, info = env.step(GBA_ACTIONS.index(control))
            action_counts[control] += 1
            if control == "A":
                battle_a_presses += 1
            frame = env.render()
            result = classifier.classify(frame)
            scene_update = gate.update(result)
            battle_score = battle_detector.score(frame)
            target_score = target_detector.score(frame)
            battle_match_scores.append(battle_score)
            target_match_scores.append(target_score)
            battle_gate.update(frame)
            target_gate.update(frame)
            if exp_gate.update(frame) and not battle_victory_exp_confirmed:
                battle_victory_exp_confirmed = True
                Image.fromarray(frame).save(out / "battle_result_earned_6_exp.png")
                events.append({"step": marker, "event": "specific_earned_6_exp_banner_confirmed",
                               "visual_anchor_score": round(exp_detector.score(frame), 5)})
                frames.append(_overlay(frame, policy=policy_label, phase="battle_victory_exp",
                                       step=int(marker), action=control, scene="result_message", battle_count=1))
                frame_labels.append(f"{marker}: specific Earned 6 EXP result")
            actions.append({"decision": marker, "action": control, "phase": phase,
                            "candidate_scene": result.label, "candidate_status": result.status,
                            "confirmed_scene": gate.confirmed_scene or "unknown",
                            "emulator_frames": info.get("emulator_frames_executed", 1),
                            "reward": float(reward)})
            capture(control, marker, phase, frame, result, scene_update)
            return frame, terminated, truncated

        frame = env.render()
        result = classifier.classify(frame)
        update = gate.update(result)
        battle_gate.update(frame)
        capture("NOOP", 0, "checkpoint", frame, result, update)

        for step_no in range(1, args.decisions + 1):
            confirmed = gate.confirmed_scene
            if controller:
                controller.observe(confirmed)
                action = controller.next_action()
                if action is None:
                    stop_reason = controller.stop_reason or "castle_excursion_complete"
                    status = "goal_reached" if controller.castle_reached else "bounded_stop"
                    break
                phase = controller.phase
            else:
                assert random_policy is not None
                action = random_policy.next_action()
                phase = "random_exploration"
            obs, reward, terminated, truncated, info = env.step(GBA_ACTIONS.index(action))
            action_counts[action] += 1
            frame = env.render()
            result = classifier.classify(frame)
            update = gate.update(result)
            battle_score = battle_detector.score(frame)
            battle_match_scores.append(battle_score)
            battle_confirmed = battle_gate.update(frame)
            actions.append({"decision": step_no, "action": action, "phase": phase,
                            "candidate_scene": result.label, "candidate_status": result.status,
                            "confirmed_scene": gate.confirmed_scene or "unknown",
                            "emulator_frames": info.get("emulator_frames_executed", 1),
                            "reward": float(reward)})
            capture(action, step_no, phase, frame, result, update)
            if battle_confirmed and not battle_entered:
                battle_entered = True
                battle_confirmation_step = step_no
                state_bytes = env.retro_env.em.get_state()
                (out / "battle_command_checkpoint_candidate.state").write_bytes(state_bytes)
                Image.fromarray(frame).save(out / "battle_command_confirmed.png")
                events.append({"step": step_no, "event": "battle_command_screen_confirmed",
                               "visual_anchor_score": round(battle_score, 5)})
                if args.attack_on_confirmed_battle:
                    current_command_ready = True
                    for actor_index in range(1, args.max_attackers + 1):
                        if not current_command_ready:
                            break
                        target_gate.reset()
                        marker = step_no + actor_index / 100
                        frame, terminated, truncated = battle_input(
                            "A", "battle_attack_select", marker)
                        if terminated or truncated:
                            stop_reason = "emulator_ended_after_attack_selection"
                            status = "emulator_terminated" if terminated else "episode_truncated"
                            break
                        target_gate.reset()
                        for offset in range(1, 13):
                            frame, terminated, truncated = battle_input(
                                "NOOP", "battle_target_wait", step_no + actor_index / 100 + offset / 1000)
                            if target_gate.confirmed or terminated or truncated:
                                break
                        Image.fromarray(frame).save(out / ("battle_command_after_A.png" if actor_index == 1
                                                            else f"battle_actor_{actor_index:02d}_target.png"))
                        if not target_gate.confirmed:
                            battle_target_confirmed = actor_index > 1
                            current_command_ready = False
                            stop_reason = "attack_selected_but_target_panel_not_confirmed"
                            status = "battle_input_attempted"
                            break
                        battle_target_confirmed = True
                        events.append({"step": step_no + actor_index / 100,
                                       "event": "single_enemy_target_panel_confirmed",
                                       "actor_slot": actor_index,
                                       "visual_anchor_score": round(target_detector.score(frame), 5)})
                        frame, terminated, truncated = battle_input(
                            "A", "battle_target_confirm", step_no + actor_index / 100 + 0.5)
                        if terminated or truncated:
                            stop_reason = "emulator_ended_after_target_confirmation"
                            status = "emulator_terminated" if terminated else "episode_truncated"
                            break
                        battle_commands += 1
                        (out / f"battle_actor_{actor_index:02d}_targeted.state").write_bytes(
                            env.retro_env.em.get_state())
                        for offset in range(2):
                            frame, terminated, truncated = battle_input(
                                "NOOP", "battle_post_target_wait",
                                step_no + actor_index / 100 + 0.6 + offset / 1000)
                            if terminated or truncated:
                                break
                        if actor_index == 1:
                            Image.fromarray(frame).save(out / "battle_after_target_A.png")
                            (out / "battle_after_target_candidate.state").write_bytes(
                                env.retro_env.em.get_state())
                        battle_gate.reset()
                        next_command_confirmed = False
                        if actor_index < args.max_attackers:
                            for wait_no in range(1, 91):
                                post_target_wait_frames = wait_no
                                frame, terminated, truncated = battle_input(
                                    "NOOP", "battle_wait_for_next_command",
                                    step_no + actor_index / 100 + 0.7 + wait_no / 1000)
                                if battle_gate.confirmed or terminated or truncated:
                                    break
                            next_command_confirmed = battle_gate.confirmed
                        if actor_index < args.max_attackers and not next_command_confirmed:
                            current_command_ready = False
                            stop_reason = f"attack_{actor_index}_targeted; next_party_command_not_confirmed"
                            status = "battle_attack_targeted"
                            break
                        if actor_index < args.max_attackers:
                            Image.fromarray(frame).save(out / ("battle_next_command_confirmed.png"
                                                               if actor_index == 1
                                                               else f"battle_actor_{actor_index + 1:02d}_command.png"))
                            events.append({"step": step_no + actor_index / 100 + 0.7 + post_target_wait_frames / 1000,
                                           "event": "next_party_command_panel_confirmed",
                                           "actor_slot": actor_index + 1,
                                           "visual_anchor_score": round(battle_detector.score(frame), 5)})
                        else:
                            stop_reason = f"{battle_commands}_party_attack_targets_confirmed; outcome_unverified"
                            status = "battle_attacks_queued"
                            next_command_confirmed = False
                    if battle_commands and status not in ("battle_input_attempted", "emulator_terminated", "episode_truncated"):
                        if battle_commands < args.max_attackers:
                            status = "battle_next_command_ready" if next_command_confirmed else "battle_attack_targeted"
                            stop_reason = f"{battle_commands}_attack_targets_confirmed; next_party_command_ready={next_command_confirmed}; outcome_unverified"
                    if battle_commands >= args.max_attackers and status == "battle_attacks_queued":
                        (out / "battle_after_full_party_candidate.state").write_bytes(
                            env.retro_env.em.get_state())
                        battle_gate.reset()
                        exterior_streak = 0
                        # Let the queued turn play without issuing more keys.
                        # A returning menu/field is reported as a scene only;
                        # no result is called victory without a result label.
                        for wait_no in range(1, args.resolution_wait_frames + 1):
                            battle_resolution_wait_frames = wait_no
                            frame, terminated, truncated = battle_input(
                                "NOOP", "battle_resolution_wait",
                                step_no + 0.8 + wait_no / 1000)
                            if battle_victory_exp_confirmed:
                                break
                            result = classifier.classify(frame)
                            exterior_streak = exterior_streak + 1 if (
                                result.label == "cornelia_exterior_field" and result.status == "candidate"
                            ) else 0
                            if battle_gate.confirmed:
                                battle_command_panel_returned = True
                                Image.fromarray(frame).save(out / "battle_command_panel_returned.png")
                                events.append({"step": step_no + 0.8 + wait_no / 1000,
                                               "event": "battle_command_panel_returned_after_wait",
                                               "visual_anchor_score": round(battle_detector.score(frame), 5)})
                                break
                            if exterior_streak >= 2:
                                field_return_after_battle = True
                                Image.fromarray(frame).save(out / "battle_field_return_candidate.png")
                                events.append({"step": step_no + 0.8 + wait_no / 1000,
                                               "event": "exterior_field_anchor_return_after_battle"})
                                break
                            if terminated or truncated:
                                break
                        if battle_victory_exp_confirmed and not terminated and not truncated:
                            # Acknowledge the explicit EXP message once, then
                            # require the exact observed Gil follow-up before
                            # pressing anything else.
                            result_marker = step_no + 0.9
                            frame, terminated, truncated = battle_input(
                                "A", "acknowledge_exp_award", result_marker)
                            gil_gate.reset()
                            for result_wait in range(1, 121):
                                frame, terminated, truncated = battle_input(
                                    "NOOP", "wait_for_gil_award",
                                    result_marker + result_wait / 1000)
                                if gil_gate.update(frame):
                                    battle_gil_award_confirmed = True
                                    Image.fromarray(frame).save(out / "battle_result_obtained_24_gil.png")
                                    events.append({"step": result_marker + result_wait / 1000,
                                                   "event": "specific_obtained_24_gil_banner_confirmed",
                                                   "visual_anchor_score": round(gil_detector.score(frame), 5)})
                                    frames.append(_overlay(frame, policy=policy_label, phase="battle_gil_award",
                                        step=int(result_marker + result_wait / 1000), action="NOOP",
                                        scene="result_message", battle_count=1))
                                    frame_labels.append("specific Obtained 24 gil result")
                                    break
                                if terminated or truncated:
                                    break
                            if battle_gil_award_confirmed and not terminated and not truncated:
                                map_gate.reset()
                                for ack_attempt in range(1, 4):
                                    frame, terminated, truncated = battle_input(
                                        "A", "acknowledge_gil_award",
                                        result_marker + 0.99 + ack_attempt / 10000)
                                    if terminated or truncated:
                                        break
                                    for map_wait in range(1, 61):
                                        frame, terminated, truncated = battle_input(
                                            "NOOP", "wait_for_postbattle_map",
                                            result_marker + 1.0 + ack_attempt / 100 + map_wait / 1000)
                                        if map_gate.update(frame):
                                            post_battle_map_view_confirmed = True
                                            Image.fromarray(frame).save(out / "battle_postbattle_map_view.png")
                                            events.append({"step": result_marker + 1.0 + ack_attempt / 100 + map_wait / 1000,
                                                           "event": "two_frame_cornelia_postbattle_map_view_confirmed",
                                                           "visual_anchor_score": round(map_detector.score(frame), 5)})
                                            frames.append(_overlay(frame, policy=policy_label,
                                                phase="post_battle_map_view_confirmed", step=int(result_marker + map_wait),
                                                action="NOOP", scene="narrow_cornelia_map_view", battle_count=1))
                                            frame_labels.append("post-battle Cornelia map view (narrow anchor)")
                                            break
                                        if terminated or truncated:
                                            break
                                    if post_battle_map_view_confirmed or terminated or truncated:
                                        break
                                    # Retry only while the exact known Gil
                                    # banner remains; otherwise stop safely.
                                    if not gil_detector.is_candidate(frame):
                                        break
                        if field_return_after_battle:
                            status = "battle_exited_outcome_unknown"
                            stop_reason = "field_anchor_returned_after_commands; battle_result_unverified"
                        elif battle_command_panel_returned:
                            status = "battle_command_panel_returned"
                            stop_reason = "battle_command_panel_returned; result_unverified"
                        elif post_battle_map_view_confirmed:
                            status = "battle_victory_and_map_resume_confirmed"
                            stop_reason = "6_EXP_and_24_GIL_confirmed_then_two_frame_Cornelia_map_anchor_returned"
                        elif battle_victory_exp_confirmed:
                            status = "battle_exp_award_confirmed"
                            stop_reason = "exact_reviewed_earned_6_exp_banner_confirmed; specific_encounter_victory"
                        else:
                            status = "battle_resolution_unknown"
                            stop_reason = f"no_confirmed_result_within_{args.resolution_wait_frames}_neutral_frames"
                    else:
                        stop_reason = "attack_selected_but_target_panel_not_confirmed"
                        status = "battle_input_attempted"
                else:
                    stop_reason = "battle_command_screen_confirmed"
                    status = "battle_observed"
                break
            if controller:
                controller.observe(gate.confirmed_scene)
                if controller.phase == "complete":
                    status = "goal_reached"
                    stop_reason = "castle_entry_and_exterior_return_confirmed"
                    break
                if controller.stop_reason:
                    status = "bounded_stop"
                    stop_reason = controller.stop_reason
                    break
                if gate.confirmed_scene in ("menu", "cornelia_town"):
                    status = "unexpected_scene"
                    stop_reason = f"unexpected_confirmed_scene:{gate.confirmed_scene}"
                    break
            if terminated or truncated:
                status = "emulator_terminated" if terminated else "episode_truncated"
                stop_reason = status
                break

        if status == "running":
            status = "bounded_run_complete"
            stop_reason = "decision_limit"

        # Preserve a final actual frame and build the video even after a
        # bounded controller stop. GIF export uses Pillow only.
        if env.render() is not None and len(frames) < 2:
            frames.append(_overlay(env.render(), policy=policy_label, phase=stop_reason,
                                   step=len(actions), action="NOOP", scene=gate.confirmed_scene or "unknown",
                                   battle_count=0))
            frame_labels.append(f"final: {stop_reason}")
        _write_gif(frames, out / "autoplay.gif")
        _contact_sheet(frames, out / "contact_sheet.png", frame_labels)

    except Exception as exc:
        status = "failed"
        stop_reason = f"{type(exc).__name__}: {exc}"
        (out / "exception.txt").write_text(stop_reason + "\n", encoding="utf-8")
        if env is not None and env.render() is not None and not frames:
            frames.append(_overlay(env.render(), policy=policy_label, phase="failure",
                                   step=0, action="NOOP", scene=gate.confirmed_scene or "unknown",
                                   battle_count=0))
            frame_labels.append("failure frame")
            _write_gif(frames, out / "autoplay.gif")
            _contact_sheet(frames, out / "contact_sheet.png", frame_labels)
    finally:
        if env is not None:
            env.close()

    elapsed = time.perf_counter() - started
    metrics = {
        "run_id": run_id, "status": status, "stop_reason": stop_reason,
        "policy": args.policy, "policy_is_learned": False, "seed": args.seed,
        "checkpoint_sha256": checkpoint_hash,
        "reset_observation_sha256": reset_info.get("observation_sha256"),
        "repeat_reset_observation_hashes_match": len(reset_hashes) == 2 and reset_hashes[0] == reset_hashes[1],
        "reset_checkpoint_loaded": bool(reset_info.get("checkpoint_loaded")),
        "reset_scene_confirmed": initial_scene if status != "failed" or reset_info else None,
        "decisions": len(actions), "emulator_frames": sum(row["emulator_frames"] for row in actions),
        "action_counts": action_counts, "confirmed_scenes": confirmed_scenes,
        "castle_reached": bool(controller and controller.castle_reached),
        "field_return_confirmed": bool(controller and controller.field_return_confirmed),
        "battle_command_screen_confirmed": battle_entered,
        "battle_target_confirmed": battle_target_confirmed,
        "battle_A_presses": battle_a_presses,
        "next_party_command_confirmed": next_command_confirmed,
        "post_target_wait_frames": post_target_wait_frames,
        "battle_command_panel_returned": battle_command_panel_returned,
        "field_return_after_battle": field_return_after_battle,
        "specific_obtained_24_gil_banner_confirmed": battle_gil_award_confirmed,
        "post_battle_map_view_confirmed": post_battle_map_view_confirmed,
        "battle_resolution_wait_frames": battle_resolution_wait_frames,
        "battle_target_anchor_scores": [round(value, 5) for value in target_match_scores],
        "battle_confirmation_step": battle_confirmation_step,
        "battle_anchor_scores_min_max": ([round(min(battle_match_scores), 5), round(max(battle_match_scores), 5)]
                                          if battle_match_scores else None),
        "battle_detection": "two-frame 100x50 Attack-selected left-panel anchor; currently validated on Goblin trials",
        "battle_commands": battle_commands,
        "battle_outcomes": {
            "victory": int(battle_victory_exp_confirmed),
            "escape": 0, "defeat": 0,
            "unknown": int(battle_entered and not battle_victory_exp_confirmed),
        },
        "specific_earned_6_exp_banner_confirmed": battle_victory_exp_confirmed,
        "unknown_scene_observations": unknown_count,
        "events": events, "elapsed_seconds": round(elapsed, 3), "training_timesteps": 0,
        "output_directory_local_only": str(out.relative_to(ROOT)),
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (out / "actions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("decision", "action", "phase", "candidate_scene",
            "candidate_status", "confirmed_scene", "emulator_frames", "reward"))
        writer.writeheader()
        writer.writerows(actions)
    report = "\n".join([
        "# GBA autonomous gameplay recording",
        f"Run ID: {run_id}", f"Status: {status}", f"Policy: {args.policy} (scripted/random; not learned)",
        f"Stop reason: {stop_reason}", f"Decisions: {len(actions)}; emulator frames: {metrics['emulator_frames']}",
        f"Confirmed scenes: {', '.join(confirmed_scenes) if confirmed_scenes else 'none'}",
        f"Castle reached: {metrics['castle_reached']}; field return confirmed: {metrics['field_return_confirmed']}",
        f"Battle command screen confirmed: {battle_entered} at decision {battle_confirmation_step}; strict two-frame 100x50 left-menu match, Goblin trials only.",
        f"Attack command sequences: {battle_commands}; target panel confirmed: {battle_target_confirmed}; A presses: {battle_a_presses}.",
        f"Specific Earned 6 EXP result banner: {battle_victory_exp_confirmed} (only this observed Goblin reward is verified).",
        f"Specific Obtained 24 gil banner: {battle_gil_award_confirmed}; narrow Cornelia-area map view after acknowledgement: {post_battle_map_view_confirmed}.",
        f"Next party command panel confirmed: {next_command_confirmed} after {post_target_wait_frames} neutral frames.",
        f"After full party inputs: command panel returned={battle_command_panel_returned}; field anchor returned={field_return_after_battle}; neutral wait={battle_resolution_wait_frames} frames.",
        ("The exact reviewed Earned 6 EXP banner independently confirms a victory in this one Goblin encounter."
         if battle_victory_exp_confirmed else
         ("No battle command screen was confirmed; outcome not applicable."
          if not battle_entered else
          "Turn resolution and battle outcome remain unknown. Leaving the battle screen is not treated as victory.")),
        f"Unknown scene observations: {unknown_count}", f"Runtime: {elapsed:.3f} seconds",
        "GIF frames are captures from the Stable-Retro emulator with an on-screen policy/scene overlay.",
        "No PPO/optimizer training ran; reward remained neutral.", "",
    ])
    (out / "run_report.txt").write_text(report, encoding="utf-8")
    if args.publish:
        public_dir = _public_package(out, metrics, report)
        print(f"Sanitized evidence: {public_dir.relative_to(ROOT)}")
    print(f"GBA autoplay output: {out}")
    print(f"GIF recording: {out / 'autoplay.gif'}")
    print(f"Metrics: {metrics_path}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("scripted_baseline", "seeded_random"), default="scripted_baseline")
    parser.add_argument("--checkpoint", help="Explicit local native checkpoint; defaults to verified Seed-party state")
    parser.add_argument("--decisions", type=int, default=280, help="Maximum one-emulator-frame actions")
    parser.add_argument("--max-approach", type=int, default=160)
    parser.add_argument("--max-return", type=int, default=112)
    parser.add_argument("--max-attackers", type=int, choices=range(1, 5), default=4,
                        help="Bound attack/target selections to this many party slots")
    parser.add_argument("--resolution-wait-frames", type=int, choices=range(1, 2401), default=1200,
                        help="Bound neutral frames to observe after all party attacks")
    parser.add_argument("--capture-every", type=int, default=2, help="Capture a GIF frame every N actions")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--attack-on-confirmed-battle", action="store_true",
                        help="After two battle-panel matches, issue bounded Attack/target confirmations and observe for the exact reviewed EXP result")
    parser.add_argument("--dry-run", action="store_true", help="Validate dependencies, checkpoint, and anchors only")
    parser.add_argument("--publish", action="store_true", help="Also write sanitized metrics/report/contact sheet under docs/")
    args = parser.parse_args(argv)
    if min(args.decisions, args.capture_every, args.max_approach, args.max_return,
           args.max_attackers, args.resolution_wait_frames) < 1:
        parser.error("decision, capture interval, and route limits must be positive")
    try:
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else discover_checkpoint()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if args.dry_run:
            import numpy  # noqa: F401
            import stable_retro
            from PIL import Image  # noqa: F401
            from ff1_dawn_ai.gba_env import FF1GbaEnv  # noqa: F401
            from scripts.gba_castle_learning_smoke import FIELD_VARIANTS, REFERENCES
            stable_retro.data.Integrations.add_custom_path(str(ROOT / "custom_integrations"))
            game = "FinalFantasyDawnOfSouls-GbAdvance"
            if game not in stable_retro.data.list_games(stable_retro.data.Integrations.ALL):
                raise RuntimeError(f"Stable-Retro integration missing: {game}")
            missing = [str(path) for path in (*REFERENCES.values(), *FIELD_VARIANTS,
                                               ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/attack_command_positive_left.png")
                       + (ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/goblin_target_positive_left.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_ui_reference/magic_menu_negative_left.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/earned_6_exp_banner.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/obtained_24_gil_banner.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_static_anchor.png")
                       if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError("Missing reviewed scene assets: " + ", ".join(missing))
            print(f"Dry-run passed. Official checkpoint exists ({checkpoint.stat().st_size} bytes); no emulator started.")
            return 0
        if args.policy == "scripted_baseline" and args.decisions < args.max_approach + args.max_return:
            print("Note: decision limit is shorter than the full bounded round trip.")
        out = run(args)
        completed = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        return 1 if completed.get("status") == "failed" else 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
