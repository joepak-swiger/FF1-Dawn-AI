from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_project_storage import run_audit  # noqa: E402


class StorageAuditTests(unittest.TestCase):
    def test_inventory_reports_git_status_and_duplicate_hash_without_deleting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
            (root / "tracked.txt").write_text("tracked", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", ".gitignore", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=test", "-c",
                            "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
            (root / "untracked.txt").write_text("other", encoding="utf-8")
            (root / "copy.txt").write_text("other", encoding="utf-8")
            (root / "ignored.bin").write_bytes(b"private fixture")
            cache = root / "scripts" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "sample.pyc").write_bytes(b"regenerable")

            output = run_audit(root, root / "reports", hash_budget_mib=1)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["git_status_breakdown"]["tracked"]["files"], 2)
            self.assertEqual(summary["git_status_breakdown"]["ignored"]["files"], 1)
            self.assertEqual(summary["git_status_breakdown"]["untracked"]["files"], 3)
            self.assertTrue(summary["duplicate_groups"])
            self.assertTrue(summary["oldest_30_files"])
            self.assertTrue(summary["newest_30_files"])
            self.assertTrue(cache.exists())

    def test_safe_clean_only_removes_source_tree_python_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
            (root / "important.state").write_bytes(b"keep")
            cache = root / "pkg" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "module.pyc").write_bytes(b"cache")
            protected = root / ".venv" / "Lib" / "__pycache__"
            protected.mkdir(parents=True)
            (protected / "module.pyc").write_bytes(b"venv-cache")

            output = run_audit(root, root / "reports", safe_clean=True)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(cache.exists())
            self.assertTrue(protected.exists())
            self.assertTrue((root / "important.state").exists())
            self.assertGreater(summary["safe_cache_bytes_removed"], 0)
            self.assertEqual(summary["bytes_after_cleanup_estimated"],
                             summary["bytes_before_cleanup"] - summary["safe_cache_bytes_removed"])


if __name__ == "__main__":
    unittest.main()
