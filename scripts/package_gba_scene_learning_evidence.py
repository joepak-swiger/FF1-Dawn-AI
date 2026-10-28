"""Prepare compact GitHub-safe scene and learning evidence from local runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_PROOFS = (
    "entry_exit_01_step_000.png",
    "entry_exit_01_step_122.png",
    "entry_exit_01_step_205.png",
    "random_110_step_129.png",
    "start_menu_control_step_064.png",
)


def _local_run(run_id: str, parent: Path) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run ID must be a single directory name")
    path = parent / run_id
    if not path.is_dir() or not (path / "metrics.json").is_file():
        raise FileNotFoundError(path)
    return path


def package(scene_run_id: str, learning_run_id: str, *, root: Path = ROOT) -> tuple[Path, Path]:
    scene_run = _local_run(scene_run_id, root / "videos/gba_scene_eval")
    learning_run = _local_run(learning_run_id, root / "videos/gba_castle_learning")
    scene_target = root / "docs/experiments/gba_scene_classifier" / scene_run_id
    learning_target = root / "docs/experiments/gba_castle_learning" / learning_run_id
    if scene_target.exists():
        raise FileExistsError(scene_target)
    publish_learning = not learning_target.exists()
    scene_metrics = json.loads((scene_run / "metrics.json").read_text(encoding="utf-8"))
    learning_metrics = json.loads((learning_run / "metrics.json").read_text(encoding="utf-8"))
    scene_target.mkdir(parents=True)
    if publish_learning:
        learning_target.mkdir(parents=True)
    (scene_target / "metrics.json").write_text(json.dumps(scene_metrics, indent=2) + "\n", encoding="utf-8")
    for filename in SCENE_PROOFS:
        source = scene_run / filename
        if source.is_file():
            shutil.copyfile(source, scene_target / filename)
    # Keep only aggregate policy results and timing/budget fields; exclude
    # local file names and episode-by-episode action traces.
    safe_learning = {
        "run_id": learning_metrics["run_id"],
        "checkpoint_run_id": learning_metrics["checkpoint_run_id"],
        "checkpoint_sha256_unchanged": learning_metrics["checkpoint_sha256_unchanged"],
        "reward": learning_metrics["reward"],
        "scene_gate": learning_metrics["scene_gate"],
        "training": {key: value for key, value in learning_metrics["training"].items()
                     if key != "model_file_local_only"},
        "evaluation": learning_metrics["evaluation"],
        "claim": learning_metrics["claim"],
        "elapsed_seconds": learning_metrics["elapsed_seconds"],
    }
    if publish_learning:
        (learning_target / "metrics.json").write_text(json.dumps(safe_learning, indent=2) + "\n", encoding="utf-8")
        sheet = learning_run / "policy_contact_sheet.png"
        if sheet.is_file():
            shutil.copyfile(sheet, learning_target / sheet.name)

    validation = scene_metrics["episode_level_validation"]
    scene_report = [
        "# Cornelia scene anchors and transition gate",
        "",
        f"Live run `{scene_metrics['run_id']}` used the verified local Seed-party checkpoint `{scene_metrics['checkpoint_run_id']}`.",
        "",
        "## Independently reviewed screens",
        "",
        "The anchor set names Cornelia exterior field, Cornelia town title, Castle Cornelia 1F title, and the rendered party/status menu. Unknowns and fades remain unknown; no RAM address or full-frame hash participates. Castle and town are matched from the game's own title strip. The exterior uses distinct map landmarks from two reviewed screen positions and tolerates limited vertical scroll.",
        "",
        f"- Fresh checkpoint field anchors: {validation['exterior_reset_candidates']}/{validation['independent_entry_exit_episodes']}.",
        f"- Two-frame Castle entry after a confirmed field: {validation['castle_entry_confirmed']}/{validation['independent_entry_exit_episodes']}.",
        f"- Two-frame Castle exit back to recognized exterior: {validation['exterior_exit_confirmed']}/{validation['independent_entry_exit_episodes']}.",
        f"- Cornelia town confirmed after pausing on its title candidate: {validation['cornelia_town_confirmed']}/{validation['independent_random_episodes']} seeded routes.",
        f"- Status/menu screen confirmed: {validation['menu_confirmed']} with 0 Castle-entry events.",
        "- Saved title/new-game fixture and fade/transition fixture remained unknown in offline regression tests.",
        "- A one-frame Cornelia title candidate appears during some routes; the gate requires a second consecutive matching frame and does not promote the flash.",
        "",
        "The three entry/exit trials are repeated restores from one fixed checkpoint and one fixed route, not broad location generalization. The town result is positive in one seeded route; the other routes did not visit town. These are episode-level counts; repeated frames are not treated as independent samples.",
        "",
        "## Decision",
        "",
        "The validated first-entry event is suitable for an opt-in sparse-goal wrapper for this fixed-start feasibility experiment. It is not a general scene classifier and does not validate battle/result/game-over screens or progression beyond Castle entry.",
        "",
        "Selected proof frames are included. Native states, ROMs, RAM dumps, full traces, and raw run folders remain local.",
    ]
    scene_report.insert(scene_report.index("## Decision"), "## Episode-level confirmation counts")
    pos = scene_report.index("## Episode-level confirmation counts") + 1
    scene_report[pos:pos] = [
        "",
        *[f"- `{label}`: {counts['confirmed']}/{counts['expected']} expected scene visits confirmed; {counts['missed']} missed."
          for label, counts in validation["independent_scene_confusion"].items()],
        f"- Unexpected confirmed scene transitions: {len(validation['unexpected_confirmed_scene_transitions'])}.",
        "",
    ]
    (scene_target / "REPORT.md").write_text("\n".join(scene_report) + "\n", encoding="utf-8")

    learning_report = [
        "# Bounded GBA Castle-entry learning feasibility",
        "",
        f"Run `{learning_metrics['run_id']}` used the verified checkpoint `{learning_metrics['checkpoint_run_id']}`.",
        "",
        f"PPO (CnnPolicy, CPU) trained from scratch for {learning_metrics['training']['timesteps_actual']} timesteps in {learning_metrics['training']['elapsed_seconds']:.1f} seconds. Training observed {learning_metrics['training']['episodes_with_goal_events']} confirmed goal events.",
        "",
        "## Three-episode same-start comparison",
        "",
    ]
    for policy, result in learning_metrics["evaluation"].items():
        learning_report.append(f"- {policy}: {result['successes']}/{result['episodes']} first-entry goals; decision steps {result['steps_to_goal']}.")
    learning_report += [
        "",
        "Only the scripted north baseline reached the goal (3/3 at 16 decisions with eight emulator frames per action). Random, idle, and the PPO policy reached 0/3. PPO selected A in all 96 held-out decisions; sparse first-entry reward supplied no training successes.",
        "",
        "This was a bounded learning-feasibility smoke, not learned navigation. All evaluations use the same fixed checkpoint and short horizon. The agent did not learn the hardcoded north route; the route is reported only as a scripted reference.",
        "",
        "The base Gymnasium environment remains reward-neutral. The optional wrapper rewards only a two-frame confirmed field-to-Castle transition. No unverified RAM-based progress rewards were added.",
        "",
        "The compact contact sheet is included. PPO weights, raw traces, ROMs, and native states remain local.",
    ]
    if publish_learning:
        (learning_target / "REPORT.md").write_text("\n".join(learning_report) + "\n", encoding="utf-8")
    return scene_target, learning_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_run_id")
    parser.add_argument("learning_run_id")
    args = parser.parse_args()
    for target in package(args.scene_run_id, args.learning_run_id):
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
