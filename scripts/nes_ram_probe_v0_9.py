from pathlib import Path
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from ff1_dawn_ai.nes_env import FF1NesEnv
from ff1_dawn_ai.ram_analysis import pixel_scene_features


OUT_DIR = PROJECT_ROOT / "videos" / "nes_ram_probe_v0_9"
FRAME_DIR = OUT_DIR / "frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

PROBE_STEPS = 900
FRAME_EVERY = 25

MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "ff1_nes_ppo_v0_7_best_reported.zip",
    PROJECT_ROOT / "models" / "ff1_nes_ppo_v0_6_best_continue.zip",
    PROJECT_ROOT / "models" / "ff1_nes_ppo_v0_5_best_quality.zip",
]


def log(message=""):
    print(message)


def make_env():
    return FF1NesEnv(
        project_root=PROJECT_ROOT,
        max_steps=PROBE_STEPS + 5,
        action_freq=16,
        print_buttons=False,
    )


def read_ram(env):
    """
    Try several common emulator RAM access paths.

    Different Retro/Gym/Stable-Retro wrappers expose RAM in slightly different places,
    so this probes safely and records which path worked.
    """
    objects_to_try = [
        ("env", env),
        ("env.retro_env", getattr(env, "retro_env", None)),
        ("env.retro_env.unwrapped", getattr(getattr(env, "retro_env", None), "unwrapped", None)),
        ("env.retro_env.env", getattr(getattr(env, "retro_env", None), "env", None)),
        ("env.retro_env.unwrapped.em", getattr(getattr(getattr(env, "retro_env", None), "unwrapped", None), "em", None)),
    ]

    for label, obj in objects_to_try:
        if obj is None:
            continue

        getter = getattr(obj, "get_ram", None)
        if callable(getter):
            try:
                ram = np.asarray(getter(), dtype=np.uint8).reshape(-1)
                if ram.size > 0:
                    return ram, f"{label}.get_ram()"
            except Exception:
                pass

        ram_attr = getattr(obj, "ram", None)
        if ram_attr is not None:
            try:
                ram = np.asarray(ram_attr, dtype=np.uint8).reshape(-1)
                if ram.size > 0:
                    return ram, f"{label}.ram"
            except Exception:
                pass

    raise RuntimeError(
        "Could not read RAM from this Stable-Retro environment. "
        "The report folder should still explain what failed."
    )


def battle_features(rgb):
    """
    Same rough visual heuristic used in v0.8.

    This is only used as a temporary label for finding RAM candidates.
    The whole point of v0.9 is to move beyond trusting this heuristic.
    """
    return pixel_scene_features(rgb)


def screen_hash(rgb):
    small = np.asarray(rgb)[::8, ::8, :3]
    return hashlib.md5(small.tobytes()).hexdigest()[:12]


def prepare_obs_for_model(model, obs):
    obs = np.asarray(obs)
    policy_shape = model.policy.observation_space.shape

    if obs.shape == policy_shape:
        return obs

    if (
        len(obs.shape) == 3
        and len(policy_shape) == 3
        and obs.shape[0] == policy_shape[1]
        and obs.shape[1] == policy_shape[2]
        and obs.shape[2] == policy_shape[0]
    ):
        return np.transpose(obs, (2, 0, 1))

    return obs


def choose_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def choose_action(model, obs, env, rng):
    if model is None:
        return int(rng.randrange(len(env.action_names))), "random"

    model_obs = prepare_obs_for_model(model, obs)
    action, _state = model.predict(model_obs, deterministic=False)
    return int(np.asarray(action).item()), "model_sample"


def top_value(values):
    if len(values) == 0:
        return None, 0.0

    counts = Counter(int(v) for v in values)
    value, count = counts.most_common(1)[0]
    return value, count / len(values)


def analyze_ram(rams, battle_mask):
    rams = np.asarray(rams, dtype=np.uint8)
    battle_mask = np.asarray(battle_mask, dtype=bool)

    ram_size = rams.shape[1]
    changed_counts = np.count_nonzero(np.diff(rams, axis=0) != 0, axis=0)

    candidates = []
    changing = []

    battle_values_exist = int(np.sum(battle_mask)) > 0
    field_values_exist = int(np.sum(~battle_mask)) > 0

    for address in range(ram_size):
        values = rams[:, address]
        unique_values = sorted(int(v) for v in np.unique(values))
        unique_count = len(unique_values)

        change_count = int(changed_counts[address])
        changing.append({
            "address_dec": address,
            "address_hex": f"0x{address:04X}",
            "unique_values": unique_count,
            "changed_count": change_count,
            "first_value": int(values[0]),
            "last_value": int(values[-1]),
            "values_seen": " ".join(str(v) for v in unique_values[:32]),
        })

        if not battle_values_exist or not field_values_exist:
            continue

        battle_vals = values[battle_mask]
        field_vals = values[~battle_mask]

        battle_mean = float(np.mean(battle_vals))
        field_mean = float(np.mean(field_vals))
        all_std = float(np.std(values)) or 1.0

        battle_top, battle_top_frac = top_value(battle_vals)
        field_top, field_top_frac = top_value(field_vals)

        top_values_differ = battle_top is not None and field_top is not None and battle_top != field_top

        mean_separation = abs(battle_mean - field_mean) / all_std
        purity_bonus = battle_top_frac + field_top_frac
        different_top_bonus = 1.0 if top_values_differ else 0.0
        low_unique_bonus = max(0.0, (32 - unique_count) / 32)

        # This is not proof. It ranks addresses worth manually investigating.
        score = (
            mean_separation
            + purity_bonus
            + different_top_bonus
            + low_unique_bonus
            - (0.015 * max(0, unique_count - 16))
        )

        candidates.append({
            "address_dec": address,
            "address_hex": f"0x{address:04X}",
            "score": round(float(score), 5),
            "unique_values": unique_count,
            "changed_count": change_count,
            "battle_mean": round(battle_mean, 4),
            "field_mean": round(field_mean, 4),
            "battle_top_value": battle_top,
            "battle_top_fraction": round(float(battle_top_frac), 4),
            "field_top_value": field_top,
            "field_top_fraction": round(float(field_top_frac), 4),
            "top_values_differ": top_values_differ,
            "values_seen": " ".join(str(v) for v in unique_values[:32]),
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    changing.sort(key=lambda item: (item["changed_count"], item["unique_values"]), reverse=True)

    return candidates, changing


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    created_at = datetime.now().isoformat(timespec="seconds")
    rng = random.Random(777)

    log()
    log("FF1 NES RAM probe v0.9")
    log("----------------------")
    log("This does not train.")
    log("It records RAM while the current selected model plays.")
    log()

    env = make_env()

    try:
        obs, info = env.reset()
        initial_ram, ram_method = read_ram(env)
    except Exception as exc:
        failure_path = OUT_DIR / "run_report.txt"
        failure_path.write_text(
            "\n".join([
                "FF1 NES RAM probe v0.9",
                "======================",
                "",
                f"Created: {created_at}",
                "Status: FAILED before rollout",
                f"Error: {repr(exc)}",
                "",
                "This means the current emulator wrapper did not expose RAM through the paths tried.",
                "Next step would be inspecting the Stable-Retro env object manually.",
            ]) + "\n"
        )
        raise

    model_path = choose_model()
    model = PPO.load(model_path, device="cpu") if model_path else None

    log(f"RAM method: {ram_method}")
    log(f"RAM size: {initial_ram.size}")
    log(f"Model used: {model_path if model_path else 'none, random actions'}")
    log(f"Output folder: {OUT_DIR}")
    log()

    rows = []
    rams = []
    battle_flags = []

    prev_action_name = "RESET"
    total_reward = 0.0

    for step in range(PROBE_STEPS + 1):
        rgb = env.render()
        ram, _method = read_ram(env)
        features = battle_features(rgb)

        rams.append(ram.copy())
        battle_flags.append(features["battle_like"])

        row = {
            "step": step,
            "prev_action": prev_action_name,
            "total_reward": round(float(total_reward), 4),
            "screen_hash": screen_hash(rgb),
            "battle_like": features["battle_like"],
            "dark_fraction": features["dark_fraction"],
            "bright_fraction": features["bright_fraction"],
            "bottom_bright_fraction": features["bottom_bright_fraction"],
            "green_fraction": features["green_fraction"],
        }
        rows.append(row)

        if step % FRAME_EVERY == 0:
            Image.fromarray(rgb).save(FRAME_DIR / f"frame_{step:04d}_{row['screen_hash']}.png")

        if step == PROBE_STEPS:
            break

        action_int, action_source = choose_action(model, obs, env, rng)
        obs, reward, terminated, truncated, info = env.step(action_int)

        total_reward += reward
        prev_action_name = info.get("button", str(action_int))

        if terminated or truncated:
            break

    env.close()

    rams_array = np.asarray(rams, dtype=np.uint8)
    battle_mask = np.asarray(battle_flags, dtype=bool)

    candidates, changing = analyze_ram(rams_array, battle_mask)

    events_path = OUT_DIR / "probe_events.csv"
    candidates_path = OUT_DIR / "battle_ram_candidates.csv"
    changing_path = OUT_DIR / "changing_ram_addresses.csv"
    metrics_path = OUT_DIR / "metrics.json"
    report_path = OUT_DIR / "run_report.txt"

    write_csv(events_path, rows)
    write_csv(candidates_path, candidates[:200])
    write_csv(changing_path, changing[:200])

    np.savez_compressed(
        OUT_DIR / "ram_samples.npz",
        rams=rams_array,
        battle_like=battle_mask,
    )

    battle_count = int(np.sum(battle_mask))
    field_count = int(np.sum(~battle_mask))
    unique_screens = len(set(row["screen_hash"] for row in rows))

    metrics = {
        "created_at": created_at,
        "purpose": "Find RAM addresses that may correspond to battle/menu/field state by correlating RAM values with visual labels.",
        "important_warning": "The battle_like labels are still image-based. Candidate RAM addresses require manual confirmation.",
        "ram_method": ram_method,
        "ram_size": int(initial_ram.size),
        "model_used": str(model_path) if model_path else None,
        "steps_recorded": len(rows),
        "unique_screen_hashes": unique_screens,
        "battle_like_steps": battle_count,
        "field_like_steps": field_count,
        "total_reward": round(float(total_reward), 4),
        "top_battle_ram_candidates": candidates[:30],
        "top_changing_ram_addresses": changing[:30],
        "files": {
            "events_csv": str(events_path),
            "battle_candidates_csv": str(candidates_path),
            "changing_addresses_csv": str(changing_path),
            "ram_samples_npz": str(OUT_DIR / "ram_samples.npz"),
            "frames_folder": str(FRAME_DIR),
            "run_report": str(report_path),
        },
    }

    metrics_path.write_text(json.dumps(metrics, indent=2))

    lines = [
        "FF1 NES RAM Probe v0.9",
        "======================",
        "",
        f"Created: {created_at}",
        f"RAM method: {ram_method}",
        f"RAM size: {initial_ram.size}",
        f"Model used: {model_path if model_path else 'none, random actions'}",
        "",
        "Rollout summary",
        "---------------",
        f"steps_recorded: {len(rows)}",
        f"unique_screen_hashes: {unique_screens}",
        f"battle_like_steps: {battle_count}",
        f"field_like_steps: {field_count}",
        f"total_reward: {round(float(total_reward), 4)}",
        "",
        "Important warning",
        "-----------------",
        "The battle_like labels are still image-based.",
        "This probe does not prove an address is the real battle flag.",
        "It ranks RAM addresses that are worth manually checking next.",
        "",
        "Top battle/menu RAM candidates",
        "------------------------------",
    ]

    for index, item in enumerate(candidates[:25], start=1):
        lines.append(
            f"{index}. {item['address_hex']} "
            f"score={item['score']} "
            f"unique={item['unique_values']} "
            f"changes={item['changed_count']} "
            f"battle_top={item['battle_top_value']}({item['battle_top_fraction']}) "
            f"field_top={item['field_top_value']}({item['field_top_fraction']}) "
            f"values={item['values_seen']}"
        )

    lines.extend([
        "",
        "Top changing RAM addresses",
        "--------------------------",
    ])

    for index, item in enumerate(changing[:25], start=1):
        lines.append(
            f"{index}. {item['address_hex']} "
            f"changes={item['changed_count']} "
            f"unique={item['unique_values']} "
            f"first={item['first_value']} "
            f"last={item['last_value']} "
            f"values={item['values_seen']}"
        )

    lines.extend([
        "",
        "Files written",
        "-------------",
        f"metrics: {metrics_path}",
        f"events_csv: {events_path}",
        f"battle_candidates_csv: {candidates_path}",
        f"changing_addresses_csv: {changing_path}",
        f"ram_samples_npz: {OUT_DIR / 'ram_samples.npz'}",
        f"frames_folder: {FRAME_DIR}",
        "",
        "Next step",
        "---------",
        "Review the top candidate addresses. Then build a manual verification script",
        "that watches those addresses while entering battle, exiting battle, opening menus,",
        "and returning to overworld.",
    ])

    report_path.write_text("\n".join(lines) + "\n")

    log("v0.9 RAM probe complete.")
    log(f"Run report: {report_path}")
    log(f"Metrics: {metrics_path}")
    log(f"Battle candidates: {candidates_path}")
    log(f"Changing RAM addresses: {changing_path}")
    log(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
