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
    AutonomousSessionCoordinator,
    BattleCommandDetector,
    BattleConfirmationGate,
    BattleExpResultDetector,
    BattleGilResultDetector,
    BattleLevelUpDetector,
    BattleTargetDetector,
    CorneliaPostBattleMapDetector,
    ScriptedCastleExcursion,
    SeededRandomExplorer,
    conservative_battle_outcome,
    compare_matched_movement_frames,
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
    draw.rectangle((0, 18, image.width, 56), fill=(0, 0, 0))
    draw.text((4, 19), f"POLICY: {policy.upper()[:30]}",
              fill=(255, 255, 255), font=font)
    draw.text((4, 31), "PARTY: SIF / MYTH / MILO / SEED",
              fill=(130, 230, 255), font=font)
    draw.text((4, 43), f"{phase.upper()[:12]} S{step} {action} {scene[:10]} B{battle_count}",
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


def _movement_probe(env, map_state: bytes, *, forward: str, frames: int,
                    out: Path, battle_id: int) -> dict:
    """Compare two replayed controls from the same post-result native state.

    This is a short diagnostic fork, not an episode reset. Every sample is
    replayed from the same map-view state, settled for two NOOP frames, then
    held for the same number of one-frame decisions. The main controller is
    restored to that state before returning.
    """
    from PIL import Image
    from ff1_dawn_ai.gba_env import GBA_ACTIONS

    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[forward]
    no_op_index = 0  # GBA_ACTIONS[0] is NOOP by construction.

    def sample(action_name: str, label: str) -> np.ndarray:
        env.retro_env.em.set_state(map_state)
        for _ in range(2):
            _obs, _reward, terminated, truncated, _info = env.step(no_op_index)
            if terminated or truncated:
                raise RuntimeError("emulator stopped while synchronizing movement probe")
        action_index = GBA_ACTIONS.index(action_name)
        for _ in range(frames):
            _obs, _reward, terminated, truncated, _info = env.step(action_index)
            if terminated or truncated:
                raise RuntimeError("emulator stopped during movement probe")
        for _ in range(2):
            _obs, _reward, terminated, truncated, _info = env.step(no_op_index)
            if terminated or truncated:
                raise RuntimeError("emulator stopped while releasing movement probe")
        frame = env.render().copy()
        Image.fromarray(frame).save(out / f"battle_{battle_id:02d}_movement_{label}.png")
        return frame

    try:
        idle_a = sample("NOOP", "idle_a")
        idle_b = sample("NOOP", "idle_b")
        forward_a = sample(forward, f"{forward.lower()}_a")
        forward_b = sample(forward, f"{forward.lower()}_b")
        opposite_a = sample(opposite, f"{opposite.lower()}_a")
        opposite_b = sample(opposite, f"{opposite.lower()}_b")
        metrics = compare_matched_movement_frames(
            idle_a, idle_b, forward_a, forward_b, opposite_a, opposite_b,
        )
        metrics.update({"forward_action": forward, "opposite_action": opposite,
                        "held_frames": frames,
                        "evidence_type": "matched screen-space pixel comparison; not tile coordinates"})
    finally:
        env.retro_env.em.set_state(map_state)
        for _ in range(2):
            _obs, _reward, terminated, truncated, _info = env.step(no_op_index)
            if terminated or truncated:
                raise RuntimeError("could not restore postbattle map state after movement probes")
    return metrics


def sanitized_public_metrics(metrics: dict) -> dict:
    """Select auditable fields for GitHub; omit local paths, traces, and states."""
    return {
        "run_id": metrics["run_id"],
        "status": metrics["status"],
        "policy": metrics["policy"],
        "policy_is_learned": False,
        "seed": metrics["seed"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "decisions": metrics["decisions"],
        "emulator_frames": metrics["emulator_frames"],
        "diagnostic_emulator_frames": metrics["diagnostic_emulator_frames"],
        "action_counts": metrics["action_counts"],
        "confirmed_scenes": metrics["confirmed_scenes"],
        "castle_reached": metrics["castle_reached"],
        "field_return_confirmed": metrics["field_return_confirmed"],
        "battle_command_screen_confirmed": metrics["battle_command_screen_confirmed"],
        "battles_entered": metrics["battles_entered"],
        "battles_finished": metrics["battles_finished"],
        "battle_victories": metrics["battle_victories"],
        "battle_escapes": metrics["battle_escapes"],
        "battle_defeats": metrics["battle_defeats"],
        "battle_unknown_outcomes": metrics["battle_unknown_outcomes"],
        "movement_resumed": metrics["movement_resumed"],
        "movement_probes": metrics["movement_probes"],
        "battle_encounters": metrics["battle_encounters"],
        "exp_awards": metrics["exp_awards"],
        "gil_awards": metrics["gil_awards"],
        "confirmed_level_ups": metrics["confirmed_level_ups"],
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
        "exp_reward_banner_confirmed": metrics["exp_reward_banner_confirmed"],
        "recognized_exp_amounts": metrics["exp_awards"],
        "recognized_gil_amounts": metrics["gil_awards"],
        "specific_earned_6_exp_banner_confirmed": metrics["specific_earned_6_exp_banner_confirmed"],
        "specific_obtained_24_gil_banner_confirmed": metrics["specific_obtained_24_gil_banner_confirmed"],
        "post_battle_map_view_confirmed": metrics["post_battle_map_view_confirmed"],
        "unknown_scene_observations": metrics["unknown_scene_observations"],
        "training_timesteps": 0,
        "elapsed_seconds": metrics["elapsed_seconds"],
        "interpretation": "scripted or seeded-random control; not a learned policy",
    }


def _public_package(out: Path, metrics: dict, report: str) -> Path:
    """Write the compact reviewer package; never copy native state or raw RAM."""
    public_dir = ROOT / "docs" / "experiments" / "gba_autoplay" / out.name
    public_dir.mkdir(parents=True, exist_ok=False)
    safe = sanitized_public_metrics(metrics)
    (public_dir / "metrics.json").write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")
    (public_dir / "REPORT.md").write_text(report, encoding="utf-8")
    for filename in ("contact_sheet.png", "battle_command_confirmed.png",
                     "battle_command_after_A.png", "battle_after_target_A.png",
                     "battle_next_command_confirmed.png", "battle_result_earned_6_exp.png",
                     "battle_result_obtained_24_gil.png", "battle_result_exp_banner.png",
                     "battle_result_gil_banner.png", "battle_postbattle_map_view.png",
                     "sif_level_up.png"):
        source = out / filename
        if source.is_file():
            (public_dir / filename).write_bytes(source.read_bytes())
    for source in sorted(out.glob("battle_*_*.png")):
        if any(token in source.name for token in
               ("command_confirmed", "result_earned_6_exp", "result_obtained_24_gil",
                "target_panel", "result_exp_banner", "result_gil_banner",
                "result_earned_4_exp", "result_earned_6_exp",
                "result_obtained_18_gil", "result_obtained_24_gil", "level_up_sif",
                "postbattle_map_view", "postbattle_unclassified", "movement_idle_a",
                "movement_left_a", "movement_right_a", "movement_up_a", "movement_down_a",
                "movement_resumed")):
            (public_dir / source.name).write_bytes(source.read_bytes())
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
    exp_detector = BattleExpResultDetector(str(exp_reference), alternative_references={
        4: str(ROOT / "docs/experiments/gba_autoplay/battle_win_reference/earned_4_exp_banner.png")},
        threshold=0.985)
    exp_gate = BattleConfirmationGate(exp_detector, consecutive_frames=2)
    gil_reference = ROOT / "docs/experiments/gba_autoplay/battle_win_reference/obtained_24_gil_banner.png"
    gil_detector = BattleGilResultDetector(str(gil_reference), alternative_references={
        18: str(ROOT / "docs/experiments/gba_autoplay/battle_win_reference/obtained_18_gil_banner.png")},
        threshold=0.985)
    gil_gate = BattleConfirmationGate(gil_detector, consecutive_frames=2)
    level_reference = ROOT / "docs/experiments/gba_autoplay/battle_win_reference/sif_level_up_banner.png"
    level_detector = BattleLevelUpDetector(str(level_reference), threshold=0.999)
    level_gate = BattleConfirmationGate(level_detector, consecutive_frames=2)
    map_references = (
        ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_static_anchor.png",
        ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_map_anchor_camera2.png",
        ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_map_anchor_camera3.png",
        ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_map_anchor_camera4.png",
    )
    map_detector = CorneliaPostBattleMapDetector(tuple(map(str, map_references)), threshold=0.995)
    map_gate = BattleConfirmationGate(map_detector, consecutive_frames=2)
    run_id, out = create_output(ROOT / "videos" / "gba_autoplay")
    started = time.perf_counter()
    policy_label = ("SCRIPTED" if args.policy == "scripted_baseline" else
                    "AUTONOMOUS SCRIPTED COMBAT + SEEDED RANDOM" if args.policy == "autonomous_session" else
                    "SEEDED RANDOM")
    controller = ScriptedCastleExcursion(args.max_approach, args.max_return) if args.policy == "scripted_baseline" else None
    random_policy = SeededRandomExplorer(args.seed) if args.policy in ("seeded_random", "autonomous_session") else None
    gate = SceneConfirmationGate(2)
    frames = []
    frame_labels = []
    actions = []
    events = []
    action_counts = {name: 0 for name in GBA_ACTIONS}
    unknown_count = 0
    battle_commands = 0
    total_battle_commands = 0
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
    session = AutonomousSessionCoordinator()
    movement_probes = []
    diagnostic_emulator_frames = 0
    env = None
    initial_scene = None
    try:
        env = FF1GbaEnv(checkpoint, project_root=ROOT,
                        max_steps=args.decisions + args.max_battles * (
                            args.resolution_wait_frames + args.movement_probe_frames * 8 + 240
                        ) + 400,
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
                                       action=action, scene=label, battle_count=session.battles_entered))
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
            if level_gate.update(frame) and session.current_battle_id is not None:
                if session.level_up_confirmed("Sif"):
                    Image.fromarray(frame).save(out / "sif_level_up.png")
                    Image.fromarray(frame).save(out / f"battle_{session.current_battle_id:02d}_level_up_sif.png")
                    events.append({"step": marker, "battle_id": session.current_battle_id,
                                   "event": "level_up_banner_confirmed", "character": "Sif",
                                   "visual_anchor_score": round(level_detector.score(frame), 5)})
            if exp_gate.update(frame) and not battle_victory_exp_confirmed:
                battle_victory_exp_confirmed = True
                exp_amount = exp_detector.recognized_amount(frame)
                session.reward_confirmed("exp", exp_amount)
                Image.fromarray(frame).save(out / "battle_result_exp_banner.png")
                if exp_amount is not None:
                    Image.fromarray(frame).save(out / f"battle_result_earned_{exp_amount}_exp.png")
                if session.current_battle_id is not None:
                    Image.fromarray(frame).save(out / f"battle_{session.current_battle_id:02d}_result_exp_banner.png")
                    if exp_amount is not None:
                        Image.fromarray(frame).save(out / f"battle_{session.current_battle_id:02d}_result_earned_{exp_amount}_exp.png")
                events.append({"step": marker, "battle_id": session.current_battle_id,
                               "event": "exp_reward_banner_confirmed", "amount": exp_amount,
                               "amount_template_known": exp_amount is not None,
                               "phrase_score": round(exp_detector.score(frame), 5),
                               "amount_scores": exp_detector.amount_scores(frame)})
                frames.append(_overlay(frame, policy=policy_label, phase="battle_exp_reward",
                                       step=int(marker), action=control, scene="result_message",
                                       battle_count=session.battles_entered))
                frame_labels.append(f"{marker}: Earned {exp_amount if exp_amount is not None else 'unknown'} EXP result")
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
                battle_commands = 0
                battle_victory_exp_confirmed = False
                battle_gil_award_confirmed = False
                post_battle_map_view_confirmed = False
                level_gate.reset()
                battle_command_panel_returned = False
                field_return_after_battle = False
                active_battle_id = session.battle_detected(step_no)
                battle_confirmation_step = step_no
                state_bytes = env.retro_env.em.get_state()
                (out / "battle_command_checkpoint_candidate.state").write_bytes(state_bytes)
                Image.fromarray(frame).save(out / "battle_command_confirmed.png")
                Image.fromarray(frame).save(out / f"battle_{active_battle_id:02d}_command_confirmed.png")
                events.append({"step": step_no, "battle_id": active_battle_id,
                               "event": "battle_detected_command_screen_confirmed",
                               "visual_anchor_score": round(battle_score, 5)})
                if args.attack_on_confirmed_battle or args.policy == "autonomous_session":
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
                        if actor_index == 1:
                            Image.fromarray(frame).save(out / f"battle_{active_battle_id:02d}_target_panel.png")
                        if not target_gate.confirmed:
                            battle_target_confirmed = actor_index > 1
                            current_command_ready = False
                            stop_reason = "attack_selected_but_target_panel_not_confirmed"
                            status = "battle_input_attempted"
                            break
                        battle_target_confirmed = True
                        events.append({"step": step_no + actor_index / 100,
                                       "battle_id": active_battle_id,
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
                        total_battle_commands += 1
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
                                           "battle_id": active_battle_id,
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
                                    gil_amount = gil_detector.recognized_amount(frame)
                                    session.reward_confirmed("gil", gil_amount)
                                    Image.fromarray(frame).save(out / "battle_result_gil_banner.png")
                                    if gil_amount is not None:
                                        Image.fromarray(frame).save(out / f"battle_result_obtained_{gil_amount}_gil.png")
                                        Image.fromarray(frame).save(out / f"battle_{active_battle_id:02d}_result_obtained_{gil_amount}_gil.png")
                                    events.append({"step": result_marker + result_wait / 1000,
                                                   "battle_id": active_battle_id,
                                                   "event": "gil_reward_banner_confirmed", "amount": gil_amount,
                                                   "amount_template_known": gil_amount is not None,
                                                   "phrase_score": round(gil_detector.score(frame), 5),
                                                   "amount_scores": gil_detector.amount_scores(frame)})
                                    frames.append(_overlay(frame, policy=policy_label, phase="battle_gil_reward",
                                        step=int(result_marker + result_wait / 1000), action="NOOP",
                                        scene="result_message", battle_count=session.battles_entered))
                                    frame_labels.append(f"Obtained {gil_amount if gil_amount is not None else 'unknown'} gil result")
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
                                            Image.fromarray(frame).save(out / f"battle_{active_battle_id:02d}_postbattle_map_view.png")
                                            events.append({"step": result_marker + 1.0 + ack_attempt / 100 + map_wait / 1000,
                                                           "event": "two_frame_cornelia_postbattle_map_view_confirmed",
                                                           "visual_anchor_score": round(map_detector.score(frame), 5)})
                                            frames.append(_overlay(frame, policy=policy_label,
                                                phase="post_battle_map_view_confirmed", step=int(result_marker + map_wait),
                                                action="NOOP", scene="narrow_cornelia_map_view",
                                                battle_count=session.battles_entered))
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
                        if battle_gil_award_confirmed and not post_battle_map_view_confirmed:
                            Image.fromarray(frame).save(
                                out / f"battle_{active_battle_id:02d}_postbattle_unclassified.png")
                            events.append({"step": step_no, "battle_id": active_battle_id,
                                           "event": "postbattle_screen_not_in_reviewed_map_anchors"})
                        outcome = "victory" if battle_victory_exp_confirmed else "unknown"
                        session.finish_battle(outcome)
                        events.append({"step": step_no, "battle_id": active_battle_id,
                                       "event": "battle_finished", "outcome": outcome,
                                       "exp_banner": battle_victory_exp_confirmed,
                                       "gil_banner": battle_gil_award_confirmed})
                        if post_battle_map_view_confirmed:
                            session.map_view_observed()
                        if field_return_after_battle:
                            status = "battle_exited_outcome_unknown"
                            stop_reason = "field_anchor_returned_after_commands; battle_result_unverified"
                        elif battle_command_panel_returned:
                            status = "battle_command_panel_returned"
                            stop_reason = "battle_command_panel_returned; result_unverified"
                        elif post_battle_map_view_confirmed:
                            if not battle_victory_exp_confirmed:
                                status = "battle_result_unknown"
                                stop_reason = "map_view_returned_without_independent_victory_evidence"
                                break
                            map_state_path = out / f"battle_{active_battle_id:02d}_postbattle_map.state"
                            map_state = env.retro_env.em.get_state()
                            map_state_path.write_bytes(map_state)
                            probe_action = random_policy.next_action() if random_policy else "UP"
                            movement = _movement_probe(env, map_state, forward=probe_action,
                                frames=args.movement_probe_frames, out=out,
                                battle_id=active_battle_id)
                            movement_probes.append({"battle_id": active_battle_id, **movement})
                            diagnostic_emulator_frames += 6 * (args.movement_probe_frames + 4) + 2
                            (out / f"battle_{active_battle_id:02d}_movement_probe.json").write_text(
                                json.dumps(movement, indent=2) + "\n", encoding="utf-8")
                            events.append({"step": step_no, "battle_id": active_battle_id,
                                           "event": "matched_idle_cardinal_movement_probe", **movement})
                            for probe_label in ("idle_a", f"{movement['forward_action'].lower()}_a",
                                                f"{movement['opposite_action'].lower()}_a"):
                                probe_path = out / f"battle_{active_battle_id:02d}_movement_{probe_label}.png"
                                if probe_path.is_file():
                                    probe_image = Image.open(probe_path).convert("RGB")
                                    frames.append(_overlay(np.asarray(probe_image), policy=policy_label,
                                        phase=f"movement_{probe_label}", step=step_no,
                                        action=probe_label, scene="postbattle_map",
                                        battle_count=session.battles_entered))
                                    frame_labels.append(f"battle {active_battle_id} matched {probe_label}")
                            if not movement["movement_signal_confirmed"]:
                                status = "postbattle_movement_unverified"
                                stop_reason = "matched_idle/cardinal probe did not show stable distinguishable screen movement"
                                events.append({"step": step_no, "battle_id": active_battle_id,
                                               "event": "movement_resume_blocked"})
                                break
                            # Replay the selected direction on the live branch
                            # after the matched-control fork, so the session
                            # truly continues without an episode reset.
                            for move_frame in range(args.movement_probe_frames):
                                obs, reward, terminated, truncated, info = env.step(
                                    GBA_ACTIONS.index(probe_action))
                                action_counts[probe_action] += 1
                                actions.append({"decision": step_no + 0.91 + move_frame / 1000,
                                    "action": probe_action, "phase": "post_battle_movement_resume",
                                    "candidate_scene": "unknown", "candidate_status": "unknown",
                                    "confirmed_scene": "unknown",
                                    "emulator_frames": info.get("emulator_frames_executed", 1),
                                    "reward": float(reward)})
                                frame = env.render()
                            for release_frame in range(2):
                                obs, reward, terminated, truncated, info = env.step(0)
                                action_counts["NOOP"] += 1
                                actions.append({"decision": step_no + 0.92 + release_frame / 1000,
                                    "action": "NOOP", "phase": "post_battle_movement_release",
                                    "candidate_scene": "unknown", "candidate_status": "unknown",
                                    "confirmed_scene": "unknown",
                                    "emulator_frames": info.get("emulator_frames_executed", 1),
                                    "reward": float(reward)})
                                frame = env.render()
                            Image.fromarray(frame).save(out / f"battle_{active_battle_id:02d}_movement_resumed.png")
                            session.movement_observed(confirmed=True)
                            events.append({"step": step_no + 0.93, "battle_id": active_battle_id,
                                "event": "post_battle_movement_resumed", "action": probe_action,
                                "verification": "repeated matched controls; screen-space pixel signal"})
                            if session.battles_entered >= args.max_battles:
                                status = "bounded_battle_cap_reached"
                                stop_reason = f"confirmed_{session.battles_finished}_battle(s); session battle cap reached"
                                break
                            # Clear all per-battle latches before resuming the
                            # same environment's field exploration loop.
                            battle_entered = False
                            battle_target_confirmed = False
                            battle_victory_exp_confirmed = False
                            battle_gil_award_confirmed = False
                            post_battle_map_view_confirmed = False
                            battle_command_panel_returned = False
                            field_return_after_battle = False
                            next_command_confirmed = False
                            battle_gate.reset()
                            target_gate.reset()
                            exp_gate.reset()
                            gil_gate.reset()
                            level_gate.reset()
                            map_gate.reset()
                            gate.reset()
                            status = "running"
                            stop_reason = "victory_acknowledged_movement_resumed_continue_exploration"
                            continue
                        elif battle_victory_exp_confirmed:
                            status = "stopped_after_confirmed_battle_unrecognized_postbattle_map"
                            stop_reason = ("specific_encounter_victory; postbattle_screen_not_in_reviewed_map_anchors"
                                           if battle_gil_award_confirmed else
                                           "earned_exp_reward_banner_confirmed; specific_encounter_victory")
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
        highlight_indices = [idx for idx, label in enumerate(frame_labels)
                             if any(term in label.lower() for term in
                                    ("battle", "exp", "gil", "movement", "map"))]
        if frames:
            highlight_indices = sorted(set([0, *highlight_indices, len(frames) - 1]))[:240]
        if highlight_indices:
            _write_gif([frames[idx] for idx in highlight_indices],
                       out / "highlights.gif", duration_ms=120)

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

    if session.current_battle_id is not None:
        current = session.encounters[session.current_battle_id - 1]
        if current["outcome"] == "pending":
            session.finish_battle("unknown")

    elapsed = time.perf_counter() - started
    any_exp = any(row["exp_confirmed"] for row in session.encounters)
    any_gil = any(row["gil_confirmed"] for row in session.encounters)
    exp_amounts = [row["exp_amount"] for row in session.encounters if row["exp_confirmed"]]
    gil_amounts = [row["gil_amount"] for row in session.encounters if row["gil_confirmed"]]
    exp_6_confirmed = 6 in exp_amounts
    gil_24_confirmed = 24 in gil_amounts
    any_map_view = any(row["map_view_observed"] for row in session.encounters)
    metrics = {
        "run_id": run_id, "status": status, "stop_reason": stop_reason,
        "policy": args.policy, "policy_is_learned": False, "seed": args.seed,
        "checkpoint_sha256": checkpoint_hash,
        "reset_observation_sha256": reset_info.get("observation_sha256"),
        "repeat_reset_observation_hashes_match": len(reset_hashes) == 2 and reset_hashes[0] == reset_hashes[1],
        "reset_checkpoint_loaded": bool(reset_info.get("checkpoint_loaded")),
        "reset_scene_confirmed": initial_scene if status != "failed" or reset_info else None,
        "decisions": len(actions),
        "emulator_frames": sum(row["emulator_frames"] for row in actions) + diagnostic_emulator_frames,
        "diagnostic_emulator_frames": diagnostic_emulator_frames,
        "action_counts": action_counts, "confirmed_scenes": confirmed_scenes,
        "castle_reached": bool(controller and controller.castle_reached),
        "field_return_confirmed": bool(controller and controller.field_return_confirmed),
        "battle_command_screen_confirmed": session.battles_entered > 0,
        "battles_entered": session.battles_entered,
        "battles_finished": session.battles_finished,
        "battle_victories": session.victories,
        "battle_escapes": session.escapes,
        "battle_defeats": session.defeats,
        "battle_unknown_outcomes": session.unknown_outcomes,
        "battle_encounters": session.encounters,
        "exp_awards": exp_amounts,
        "gil_awards": gil_amounts,
        "confirmed_level_ups": [
            {"battle_id": row["battle_id"], "character": name}
            for row in session.encounters for name in row["level_ups"]],
        "next_party_command_confirmations": sum(
            event.get("event") == "next_party_command_panel_confirmed" for event in events),
        "movement_resumed": session.movement_resumed,
        "movement_probes": movement_probes,
        "battle_target_confirmed": battle_target_confirmed,
        "battle_A_presses": battle_a_presses,
        "next_party_command_confirmed": next_command_confirmed,
        "post_target_wait_frames": post_target_wait_frames,
        "battle_command_panel_returned": battle_command_panel_returned,
        "field_return_after_battle": field_return_after_battle,
        "specific_obtained_24_gil_banner_confirmed": gil_24_confirmed,
        "post_battle_map_view_confirmed": any_map_view,
        "battle_resolution_wait_frames": battle_resolution_wait_frames,
        "battle_target_anchor_scores": [round(value, 5) for value in target_match_scores],
        "battle_confirmation_step": battle_confirmation_step,
        "battle_anchor_scores_min_max": ([round(min(battle_match_scores), 5), round(max(battle_match_scores), 5)]
                                          if battle_match_scores else None),
        "battle_detection": "two-frame 100x50 Attack-selected left-panel anchor; currently validated on Goblin trials",
        "battle_commands": total_battle_commands,
        "battle_outcomes": {
            "victory": session.victories,
            "escape": session.escapes, "defeat": session.defeats,
            "unknown": session.unknown_outcomes,
        },
        "specific_earned_6_exp_banner_confirmed": exp_6_confirmed,
        "exp_reward_banner_confirmed": any_exp,
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
    with (out / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "battle_id", "event", "details"))
        writer.writeheader()
        for event in events:
            writer.writerow({"step": event.get("step", ""),
                             "battle_id": event.get("battle_id", ""),
                             "event": event.get("event", ""),
                             "details": json.dumps({k: v for k, v in event.items()
                                                    if k not in {"step", "battle_id", "event"}},
                                                   sort_keys=True)})
    report = "\n".join([
        "# GBA autonomous gameplay recording",
        f"Run ID: {run_id}", f"Status: {status}",
        f"Policy: {args.policy} (field navigation is seeded-random; combat uses scripted Attack/target inputs; not learned)",
        f"Stop reason: {stop_reason}", f"Decisions: {len(actions)}; emulator frames: {metrics['emulator_frames']}",
        f"Confirmed scenes: {', '.join(confirmed_scenes) if confirmed_scenes else 'none'}",
        f"Castle reached: {metrics['castle_reached']}; field return confirmed: {metrics['field_return_confirmed']}",
        f"Distinct battle menu entries: {session.battles_entered}; finished: {session.battles_finished}; victories: {session.victories}; unknown outcomes: {session.unknown_outcomes}.",
        f"Attack/target command sequences: {total_battle_commands}; A presses: {battle_a_presses}; historical next-party panels confirmed: {metrics['next_party_command_confirmations']}; terminal next-command flag: {next_command_confirmed}.",
        f"Observed EXP amounts: {exp_amounts}; unrecognized amount count: {sum(amount is None for amount in exp_amounts)}.",
        f"Observed Gil amounts: {gil_amounts}; unrecognized amount count: {sum(amount is None for amount in gil_amounts)}; narrow map views: {sum(row['map_view_observed'] for row in session.encounters)}.",
        f"Known templates specifically matched 6 EXP={exp_6_confirmed}, 4 EXP={4 in exp_amounts}, 24 Gil={gil_24_confirmed}, 18 Gil={18 in gil_amounts}.",
        f"Level-up banners confirmed: {metrics['confirmed_level_ups']}.",
        f"Matched postbattle movement probes: {len(movement_probes)}; resumed movement confirmations: {session.movement_resumed}.",
        f"Next party command panel confirmed: {next_command_confirmed} after {post_target_wait_frames} neutral frames.",
        f"After full party inputs: command panel returned={battle_command_panel_returned}; field anchor returned={field_return_after_battle}; neutral wait={battle_resolution_wait_frames} frames.",
        ("A two-frame fixed Earned-[amount]-EXP phrase confirms each counted victory once; numeric amounts are only reported when a reviewed glyph template matches."
         if session.victories else
         ("No battle command screen was confirmed; outcome not applicable."
          if not session.battles_entered else
          "Battle outcome remains unknown. Leaving a battle-like screen is not treated as victory.")),
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
    parser.add_argument("--policy", choices=("scripted_baseline", "seeded_random", "autonomous_session"), default="scripted_baseline")
    parser.add_argument("--checkpoint", help="Explicit local native checkpoint; defaults to verified Seed-party state")
    parser.add_argument("--decisions", type=int, default=280, help="Maximum one-emulator-frame actions")
    parser.add_argument("--max-approach", type=int, default=160)
    parser.add_argument("--max-return", type=int, default=112)
    parser.add_argument("--max-attackers", type=int, choices=range(1, 5), default=4,
                        help="Bound attack/target selections to this many party slots")
    parser.add_argument("--resolution-wait-frames", type=int, choices=range(1, 2401), default=1200,
                        help="Bound neutral frames to observe after all party attacks")
    parser.add_argument("--max-battles", type=int, choices=range(1, 11), default=2,
                        help="Maximum distinct battle-menu entries in autonomous_session mode")
    parser.add_argument("--movement-probe-frames", type=int, choices=range(4, 121), default=24,
                        help="Held emulator frames in matched postbattle direction/idle controls")
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
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/earned_4_exp_banner.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/obtained_24_gil_banner.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/obtained_18_gil_banner.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/sif_level_up_banner.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_static_anchor.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_map_anchor_camera2.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_map_anchor_camera3.png",
                          ROOT / "docs/experiments/gba_autoplay/battle_win_reference/cornelia_postbattle_map_anchor_camera4.png")
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
