"""Conservative analysis helpers for controlled GBA position experiments.

The functions identify bytes that respond repeatably to a direction, differ
from elapsed-frame no-op controls, and return on a measured reverse route.
They do not assign a coordinate meaning to any offset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _matrix(samples: Sequence[np.ndarray], *, label: str) -> np.ndarray:
    if not samples:
        raise ValueError(f"{label} requires at least one RAM sample")
    rows = [np.asarray(row, dtype=np.uint8).reshape(-1) for row in samples]
    if len({row.size for row in rows}) != 1:
        raise ValueError(f"{label} RAM samples have inconsistent sizes")
    return np.stack(rows)


def analyze_ram_position_candidates(
    baseline: Sequence[np.ndarray],
    controls: Mapping[str, Sequence[np.ndarray]],
    *,
    min_repeats: int = 3,
) -> dict:
    """Compare matched no-op and cardinal samples at aligned elapsed frames.

    Returned addresses are only *candidate offsets*. A candidate must have a
    constant checkpoint value across repeats, a constant matched no-op value
    equal to that checkpoint value, and a constant direction-specific value
    distinct from it. Multiple directional responses increase its score.
    """
    base = _matrix(baseline, label="baseline")
    if base.shape[0] < min_repeats:
        raise ValueError(f"Need at least {min_repeats} independent repeats")
    names = ("UP", "DOWN", "LEFT", "RIGHT")
    missing = set(("NOOP", *names)) - controls.keys()
    if missing:
        raise ValueError(f"Missing matched control samples: {sorted(missing)}")
    arrays = {name: _matrix(controls[name], label=name) for name in ("NOOP", *names)}
    if any(value.shape != base.shape for value in arrays.values()):
        raise ValueError("All protocols must have the same repeats and RAM size")

    baseline_stable = np.all(base == base[0], axis=0)
    noop_stable = np.all(arrays["NOOP"] == arrays["NOOP"][0], axis=0)
    noop_matches = arrays["NOOP"][0] == base[0]
    eligible = baseline_stable & noop_stable & noop_matches
    per_direction = {}
    direction_counts = np.zeros(base.shape[1], dtype=np.uint8)
    signatures: dict[int, dict[str, int]] = {}
    for name in names:
        trial = arrays[name]
        stable = np.all(trial == trial[0], axis=0)
        changed = trial[0] != base[0]
        mask = eligible & stable & changed
        indices = np.flatnonzero(mask)
        per_direction[name] = [int(i) for i in indices]
        direction_counts += mask.astype(np.uint8)
        for i in indices:
            signatures.setdefault(int(i), {})[name] = int(trial[0, i])
    ranked = sorted(
        (i for i, values in signatures.items()
         if len(values) >= 2 and len(set(values.values())) >= 2),
        key=lambda i: (-len(signatures[i]), i),
    )
    return {
        "ram_size": int(base.shape[1]),
        "repeat_count": int(base.shape[0]),
        "baseline_stable_byte_count": int(baseline_stable.sum()),
        "matched_noop_stable_byte_count": int(eligible.sum()),
        "directional_candidate_counts": {key: len(value) for key, value in per_direction.items()},
        "candidate_offsets": [
            {
                "offset": i,
                "checkpoint_value": int(base[0, i]),
                "direction_values": signatures[i],
                "direction_count": len(signatures[i]),
                "direction_value_count": len(set(signatures[i].values())),
                "classification": "unverified_direction_correlated_byte",
            }
            for i in ranked[:200]
        ],
    }


def reversible_offsets(
    baseline: Sequence[np.ndarray],
    outbound: Sequence[np.ndarray],
    returned: Sequence[np.ndarray],
    matched_noop: Sequence[np.ndarray],
) -> list[int]:
    """Changed offsets that return exactly after a reverse route and idle control."""
    start = _matrix(baseline, label="baseline")
    out = _matrix(outbound, label="outbound")
    back = _matrix(returned, label="returned")
    idle = _matrix(matched_noop, label="matched no-op")
    if start.shape != out.shape or start.shape != back.shape or start.shape != idle.shape:
        raise ValueError("Reversal samples must have aligned dimensions")
    mask = (
        np.all(start == start[0], axis=0)
        & np.all(out == out[0], axis=0)
        & np.all(back == back[0], axis=0)
        & np.all(idle == idle[0], axis=0)
        & (start[0] != out[0])
        & (start[0] == back[0])
        & (start[0] == idle[0])
    )
    return [int(index) for index in np.flatnonzero(mask)]
