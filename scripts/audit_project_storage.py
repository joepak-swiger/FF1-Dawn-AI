#!/usr/bin/env python3
"""Inventory local checkout storage and optionally remove only Python caches.

The audit never removes experiment outputs, models, ROMs, states, datasets, or
the virtual environment. Duplicate hashes are bounded by ``--hash-budget-mib``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
        for name in names:
            path = Path(base) / name
            if not path.is_symlink() and path.is_file():
                files.append(path)
    return files


def git_sets(root: Path, relpaths: list[str]) -> tuple[set[str], set[str], set[str]]:
    tracked_run = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                 check=True, capture_output=True)
    tracked = {item.decode("utf-8", "surrogateescape")
               for item in tracked_run.stdout.split(b"\0") if item}
    status_run = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1",
                                 "--ignored=matching", "-z"], check=True, capture_output=True)
    ignored_prefixes: set[str] = set()
    untracked_prefixes: set[str] = set()
    raw = status_run.stdout.split(b"\0")
    for item in raw:
        if len(item) < 4:
            continue
        code = item[:2].decode("ascii", "replace")
        name = item[3:].decode("utf-8", "surrogateescape").replace("\\", "/")
        if code == "!!":
            ignored_prefixes.add(name)
            ignored_prefixes.add(name.rstrip("/") + "/")
        elif code == "??":
            untracked_prefixes.add(name)
            untracked_prefixes.add(name.rstrip("/") + "/")
    ignored = {path for path in relpaths if any(
        path == prefix or path.startswith(prefix) for prefix in ignored_prefixes)}
    untracked = {path for path in relpaths if path not in tracked and path not in ignored and any(
        path == prefix or path.startswith(prefix) for prefix in untracked_prefixes)}
    # Anything else is an untracked file status omitted only if Git status is
    # configured to hide it; normal repositories report all such paths.
    untracked.update(set(relpaths) - tracked - ignored - untracked)
    return tracked, untracked, ignored


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_cache_dirs(root: Path) -> list[Path]:
    result = []
    for base, dirs, _ in os.walk(root, topdown=True, followlinks=False):
        current = Path(base)
        dirs[:] = [d for d in dirs if not (current / d).is_symlink()]
        rel = current.relative_to(root)
        if rel.parts and rel.parts[0] in {".git", ".venv"}:
            dirs[:] = []
            continue
        for dirname in tuple(dirs):
            if dirname in {"__pycache__", ".pytest_cache"}:
                result.append(current / dirname)
                dirs.remove(dirname)
    return sorted(result)


def directory_size(path: Path) -> tuple[int, int]:
    files = walk_files(path)
    return len(files), sum(file.stat().st_size for file in files)


def run_audit(root: Path, output_root: Path, *, safe_clean: bool = False,
              hash_budget_mib: int = 512) -> Path:
    root = root.resolve()
    output_root = output_root.resolve()
    output = output_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    disk_before = shutil.disk_usage(root)
    files = walk_files(root)
    relpaths = [path.relative_to(root).as_posix() for path in files]
    tracked, untracked, ignored = git_sets(root, relpaths)

    records = []
    top: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    suffix = Counter()
    size_groups: dict[int, list[Path]] = defaultdict(list)
    for path, rel in zip(files, relpaths):
        stat = path.stat()
        status = ("git_metadata" if rel.startswith(".git/") else
                  "tracked" if rel in tracked else "ignored" if rel in ignored else "untracked")
        top_name = rel.split("/", 1)[0] or "."
        top[top_name][0] += 1
        top[top_name][1] += stat.st_size
        suffix[path.suffix.lower() or "[no extension]"] += 1
        records.append({"path": rel, "bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc).isoformat(), "git_status": status})
        size_groups[stat.st_size].append(path)

    hash_budget = hash_budget_mib * 1024 * 1024
    hash_used = 0
    hashes: dict[str, list[str]] = defaultdict(list)
    skipped_duplicate_size_groups = 0
    for size, paths in sorted(size_groups.items()):
        if len(paths) < 2 or size == 0:
            continue
        group_cost = size * len(paths)
        if hash_used + group_cost > hash_budget:
            skipped_duplicate_size_groups += 1
            continue
        hash_used += group_cost
        for path in paths:
            hashes[sha256(path)].append(path.relative_to(root).as_posix())
    duplicates = [{"sha256": digest, "bytes_each": (root / paths[0]).stat().st_size,
                   "copies": len(paths), "paths": paths}
                  for digest, paths in hashes.items() if len(paths) > 1]

    proposals = []
    for path in safe_cache_dirs(root):
        count, size = directory_size(path)
        proposals.append({"path": path.relative_to(root).as_posix(), "bytes": size,
                          "files": count, "purpose": "regenerable Python bytecode/test cache",
                          "regeneration": "Python recreates cache on import; pytest recreates cache on run",
                          "safe_action": "remove only with --safe-clean; otherwise retain",
                          "rollback": "not needed; cache is regenerated automatically"})

    if safe_clean:
        for proposal in proposals:
            target = root / proposal["path"]
            # Recheck the target is a cache directory under the checkout.
            if target.is_dir() and target.resolve().is_relative_to(root):
                shutil.rmtree(target)

    # Final inventory records the requested before-clean snapshot; report the
    # cleanup delta separately so the audit remains reproducible.
    with (output / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "mtime_utc", "git_status"))
        writer.writeheader()
        writer.writerows(records)
    with (output / "proposed_actions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("path", "bytes", "files", "purpose", "regeneration", "safe_action", "rollback")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(proposals)

    top_files = sorted(records, key=lambda row: row["bytes"], reverse=True)[:30]
    size_by_git_status = defaultdict(lambda: [0, 0])
    for row in records:
        cell = size_by_git_status[row["git_status"]]
        cell[0] += 1
        cell[1] += row["bytes"]
    git_dir = root / ".git"
    git_bytes = sum(p.stat().st_size for p in walk_files(git_dir)) if git_dir.exists() else 0
    disk_after = shutil.disk_usage(root)
    removed_bytes = sum(row["bytes"] for row in proposals) if safe_clean else 0
    summary = {
        "root": str(root), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count_before_cleanup": len(records), "bytes_before_cleanup": sum(r["bytes"] for r in records),
        "bytes_after_cleanup_estimated": sum(r["bytes"] for r in records) - removed_bytes,
        "safe_cache_files_removed": sum(r["files"] for r in proposals) if safe_clean else 0,
        "safe_cache_bytes_removed": removed_bytes,
        "disk_free_before_bytes": disk_before.free, "disk_free_after_bytes": disk_after.free,
        "git_object_store_bytes": git_bytes,
        "git_status_breakdown": {k: {"files": v[0], "bytes": v[1]} for k, v in size_by_git_status.items()},
        "top_level": {k: {"files": v[0], "bytes": v[1]} for k, v in
                      sorted(top.items(), key=lambda item: item[1][1], reverse=True)},
        "suffix_counts": dict(suffix.most_common()), "top_30_largest_files": top_files,
        "duplicate_hash_bytes_read": hash_used, "duplicate_groups": duplicates,
        "duplicate_size_groups_skipped_by_budget": skipped_duplicate_size_groups,
        "hash_budget_bytes": hash_budget, "cleanup_mode": "safe-clean" if safe_clean else "audit-only",
        "cleanup_proposals": proposals,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["FF1-Dawn-AI local storage audit", f"Root: {root}",
             f"Files: {len(records):,}", f"Bytes before cleanup: {summary['bytes_before_cleanup']:,}",
             f"Git object store: {git_bytes:,} bytes",
             f"Tracked / untracked / ignored: {size_by_git_status['tracked'][1]:,} / "
             f"{size_by_git_status['untracked'][1]:,} / {size_by_git_status['ignored'][1]:,} bytes",
             f"Safe cache removed: {removed_bytes:,} bytes" if safe_clean else "Cleanup: audit only",
             f"Disk free: {disk_before.free:,} -> {disk_after.free:,} bytes", "",
             "Largest top-level consumers:"]
    lines.extend(f"- {name}: {data[1]:,} bytes ({data[0]:,} files)" for name, data in
                 sorted(top.items(), key=lambda item: item[1][1], reverse=True)[:20])
    lines.extend(["", "Largest individual files:"])
    lines.extend(f"- {row['bytes']:,} bytes: {row['path']} [{row['git_status']}]" for row in top_files)
    lines.extend(["", f"Exact duplicate groups within hash budget: {len(duplicates)}",
                  "No experiment, model, ROM, save state, or recording was removed."])
    (output / "run_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--safe-clean", action="store_true", help="Remove only __pycache__ and .pytest_cache outside .venv/.git")
    parser.add_argument("--hash-budget-mib", type=int, default=16,
                        help="Maximum bytes read while hashing same-size duplicate candidates (default: 16 MiB)")
    args = parser.parse_args(argv)
    if args.hash_budget_mib < 0:
        parser.error("--hash-budget-mib cannot be negative")
    output_root = args.output_root or args.root / "reports" / "storage_audit"
    try:
        print(run_audit(args.root, output_root, safe_clean=args.safe_clean,
                        hash_budget_mib=args.hash_budget_mib))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"STORAGE AUDIT FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
