"""Auditable local-only supervised imitation for GBA combat.

Every supervised observation precedes the expert action. Encounter-parent
groups, not individual screenshots, determine train/validation/test splits.
This is not RL and does not infer wins, HP, or game progress.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ACTIONS = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "L", "R")
STAGES = ("attack_command", "target_confirm", "menu_other")
FEATURE_SIZE = (20, 20)


def feature(frame: np.ndarray, stage: str) -> np.ndarray:
    if stage not in STAGES:
        raise ValueError("unrecognized scene stage")
    pixels = np.asarray(frame, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("expected RGB observation")
    image = Image.fromarray(pixels[:, :, :3]).resize(FEATURE_SIZE, Image.Resampling.BILINEAR)
    vector = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    stage_vector = np.zeros(len(STAGES), dtype=np.float32)
    stage_vector[STAGES.index(stage)] = 1.0
    return np.concatenate([vector, stage_vector])


class CombatDecisionRecorder:
    """Append-only PNG + JSONL capture of verified PRE-action expert examples.

    group_id must be the original encounter-parent ID; replay branches must
    inherit it. Do not assign a new group ID to each replayed screenshot.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.output_dir / "decisions.jsonl"
        self.count = (sum(bool(line.strip()) for line in self.manifest.read_text(
            encoding="utf-8").splitlines()) if self.manifest.exists() else 0)

    def record(self, *, frame: np.ndarray, stage: str, action: str,
               group_id: str, encounter_id: str, source: str = "scripted_expert") -> dict:
        if action not in ACTIONS or stage not in STAGES:
            raise ValueError("unsupported action or stage")
        if source != "scripted_expert":
            raise ValueError("only verified expert decisions may be supervised labels")
        if not group_id.strip() or not encounter_id.strip():
            raise ValueError("encounter provenance missing")
        pixels = np.asarray(frame, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] < 3 or min(pixels.shape[:2]) < 16:
            raise ValueError("frame must be an RGB emulator observation")
        filename = f"decision_{self.count:07d}.png"
        path = self.frames_dir / filename
        if path.exists():
            raise FileExistsError(path)
        Image.fromarray(pixels[:, :, :3]).save(path, optimize=True)
        row = {
            "sample_id": self.count, "frame": f"frames/{filename}",
            "frame_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stage": stage, "action": action, "group_id": group_id,
            "encounter_id": encounter_id, "source": source,
            "timing": "observation_before_expert_button",
        }
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.count += 1
        return row


def load_dataset(directory: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    root = Path(directory).resolve()
    manifest = root / "decisions.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    features, labels, rows = [], [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") != "scripted_expert" or row.get("timing") != "observation_before_expert_button":
            raise ValueError("non-expert or non-preaction label")
        if row.get("action") not in ACTIONS or not row.get("group_id"):
            raise ValueError("missing action/group provenance")
        path = (root / row["frame"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("invalid training frame path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["frame_sha256"]:
            raise ValueError("frame checksum mismatch")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        features.append(feature(pixels, row["stage"]))
        labels.append(ACTIONS.index(row["action"]))
        rows.append(row)
    if not rows:
        raise ValueError("empty expert dataset")
    return np.stack(features), np.asarray(labels, dtype=np.int64), rows


def grouped_splits(rows: list[dict], *, seed: int = 2026) -> dict[str, np.ndarray]:
    groups = sorted({row["group_id"] for row in rows})
    if len(groups) < 5:
        raise ValueError("need >=5 independent encounter-parent groups")
    order = np.random.default_rng(seed).permutation(groups).tolist()
    n_test, n_val = max(1, round(len(order) * .2)), max(1, round(len(order) * .2))
    allocation = {
        "test": set(order[:n_test]),
        "validation": set(order[n_test:n_test + n_val]),
        "train": set(order[n_test + n_val:]),
    }
    splits = {
        name: np.asarray([i for i, row in enumerate(rows) if row["group_id"] in selected], dtype=np.int64)
        for name, selected in allocation.items()
    }
    if any(not len(indices) for indices in splits.values()):
        raise ValueError("empty data split")
    return splits


class CombatLinearPolicy:
    """Balanced linear softmax behavior cloner; unobserved actions are masked."""

    def __init__(self, weights: np.ndarray, bias: np.ndarray,
                 action_mask: np.ndarray | None = None):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.action_mask = (np.asarray(action_mask, dtype=bool) if action_mask is not None
                            else np.ones(len(ACTIONS), dtype=bool))
        if (self.weights.ndim != 2 or self.weights.shape[1] != len(ACTIONS)
                or self.bias.shape != (len(ACTIONS),)
                or self.action_mask.shape != (len(ACTIONS),) or not self.action_mask.any()):
            raise ValueError("incompatible model weights/action mask")

    def predict(self, frame: np.ndarray, stage: str) -> tuple[str, float]:
        vector = feature(frame, stage)
        if vector.shape[0] != self.weights.shape[0]:
            raise ValueError("feature size differs from trained model")
        logits = vector @ self.weights + self.bias
        logits[~self.action_mask] = -1e5
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        return ACTIONS[index], float(probabilities[index])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, bias=self.bias,
                            action_mask=self.action_mask, actions=np.asarray(ACTIONS),
                            stages=np.asarray(STAGES),
                            model_type="supervised_linear_bc_v1")

    @classmethod
    def load(cls, path: Path) -> "CombatLinearPolicy":
        with np.load(path, allow_pickle=False) as saved:
            if tuple(saved["actions"].tolist()) != ACTIONS or tuple(saved["stages"].tolist()) != STAGES:
                raise ValueError("saved action/stage mapping mismatch")
            if saved["model_type"].item() != "supervised_linear_bc_v1":
                raise ValueError("unknown model version")
            return cls(saved["weights"], saved["bias"], saved["action_mask"])


def train_policy(features: np.ndarray, labels: np.ndarray, splits: dict[str, np.ndarray],
                 *, epochs: int = 150, lr: float = .03, l2: float = .01
                 ) -> tuple[CombatLinearPolicy, dict]:
    train_idx, val_idx = splits["train"], splits["validation"]
    if len(train_idx) < 12 or len(val_idx) < 2:
        raise ValueError("insufficient independent supervised observations")
    if len(set(labels[train_idx].tolist())) < 2:
        raise ValueError("only one demonstrated action; not a meaningful learned policy")
    x_train, y_train = features[train_idx].astype(np.float32), labels[train_idx]
    x_val, y_val = features[val_idx].astype(np.float32), labels[val_idx]
    counts = np.bincount(y_train, minlength=len(ACTIONS))
    seen = counts > 0
    weight = np.zeros((features.shape[1], len(ACTIONS)), dtype=np.float32)
    bias = np.zeros(len(ACTIONS), dtype=np.float32)
    inverse = np.zeros(len(ACTIONS), dtype=np.float32)
    inverse[seen] = len(y_train) / (seen.sum() * counts[seen])
    sample_weights = inverse[y_train]
    best_accuracy, best_loss, best_params, best_epoch = -1., float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        scores = x_train @ weight + bias
        scores[:, ~seen] = -1e5
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_train)), y_train] -= 1.0
        probs *= (sample_weights / max(sample_weights.sum(), 1)).reshape(-1, 1)
        weight -= lr * (x_train.T @ probs + l2 * weight)
        bias -= lr * probs.sum(axis=0)
        scores = x_val @ weight + bias
        scores[:, ~seen] = -1e5
        accuracy = float(np.mean(scores.argmax(axis=1) == y_val))
        centered = scores - scores.max(axis=1, keepdims=True)
        p = np.exp(centered)
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(p[np.arange(len(y_val)), y_val], 1e-9)).mean())
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy, best_loss, best_params, best_epoch = accuracy, loss, (weight.copy(), bias.copy()), epoch
    policy = CombatLinearPolicy(*best_params, seen)
    metrics = {
        "method": "supervised_balanced_linear_imitation",
        "train_samples": int(len(train_idx)), "validation_samples": int(len(val_idx)),
        "test_samples": int(len(splits["test"])),
        "train_action_counts": {ACTIONS[i]: int(counts[i]) for i in np.flatnonzero(seen)},
        "validation_accuracy": best_accuracy, "best_epoch": best_epoch,
        "optimizer_epochs": epochs, "rl_timesteps": 0,
        "caveat": "offline action accuracy is not closed-loop gameplay success",
    }
    return policy, metrics


def train_from_directory(dataset: Path, model_path: Path, *, seed: int = 2026) -> dict:
    features, labels, rows = load_dataset(dataset)
    splits = grouped_splits(rows, seed=seed)
    policy, metrics = train_policy(features, labels, splits)
    scores = features[splits["test"]] @ policy.weights + policy.bias
    scores[:, ~policy.action_mask] = -1e5
    metrics["heldout_accuracy"] = float(np.mean(scores.argmax(axis=1) == labels[splits["test"]]))
    metrics["group_counts"] = {
        name: len({rows[i]["group_id"] for i in indices}) for name, indices in splits.items()
    }
    metrics["split_provenance"] = "all replay branches inherit original encounter group_id"
    policy.save(model_path)
    return metrics
