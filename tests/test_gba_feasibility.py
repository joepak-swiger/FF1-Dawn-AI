import struct
import json
import sys
import unittest
import uuid
import zlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from ff1_dawn_ai.gba_feasibility import inspect_file, png_chunks
from ff1_dawn_ai.gba_party_setup import (
    BOOT_ROUTE, Screen, classify_screen, looks_like_cornelia_field,
    meaningful_field_movement, name_changed, class_navigation, name_key_position,
    name_key_cursor_matches, class_picker_choice,
)
from ff1_dawn_ai.party_config import GBA_CUSTOM_PARTY, NES_SECONDARY_PARTY
from scripts.gba_feasibility_probe import (
    GAME,
    _initial_milestones,
    _merge_runtime_results,
    _repeated_pair_boundary_frame,
    _save_trace_frame,
    _summarize_party_trace,
    _state_machine_setup,
    runtime_smoke,
)
from scripts.package_gba_party_evidence import package_run


def make_chunk(name: bytes, payload: bytes) -> bytes:
    body = name + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


class GbaStateFormatTests(unittest.TestCase):
    def test_synthetic_runtime_reports_separate_milestones_and_partial_trace(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            project = Path(temp)
            integration = project / "custom_integrations" / GAME
            integration.mkdir(parents=True)
            (integration / "rom.gba").write_bytes(b"synthetic fixture")
            output = project / "run"
            output.mkdir()

            class FakeEnv:
                buttons = ["A", "B", "START", "SELECT", "DOWN"]

                def __init__(self, fail_after=None):
                    self.action_space = SimpleNamespace(shape=(len(self.buttons),))
                    self.unwrapped = self
                    self.counter = 0
                    self.fail_after = fail_after
                    self.em = SimpleNamespace(get_state=self.get_state, set_state=self.set_state)

                def observation(self):
                    return np.full((160, 240, 3), self.counter % 256, dtype=np.uint8)

                def reset(self):
                    self.counter = 0
                    return self.observation(), {}

                def step(self, _action):
                    self.counter += 1
                    if self.fail_after is not None and self.counter >= self.fail_after:
                        raise RuntimeError("synthetic input failure")
                    return self.observation(), 0.0, False, False, {}

                def get_state(self):
                    return self.counter.to_bytes(8, "little")

                def set_state(self, state):
                    self.counter = int.from_bytes(state, "little")

                def action_to_array(self, action):
                    return [np.asarray(action, dtype=np.uint8)]

                def close(self):
                    pass

            fake_env = FakeEnv()
            make_calls = []
            fake_retro = SimpleNamespace(
                data=SimpleNamespace(
                    Integrations=SimpleNamespace(ALL=0, add_custom_path=lambda _path: None)
                ),
                State=SimpleNamespace(NONE=0),
                Actions=SimpleNamespace(ALL=0),
                make=lambda **kwargs: (make_calls.append(kwargs) or fake_env),
            )
            with patch("scripts.gba_feasibility_probe.PROJECT_ROOT", project), patch.dict(sys.modules, {"stable_retro": fake_retro}):
                report = runtime_smoke(output, automate_party_setup=True)

            milestones = report["milestones"]
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["action_filter_mode"], "ALL")
            self.assertEqual(report["start_action_decode"][report["button_mapping"]["START"]], 1)
            self.assertEqual(make_calls[0]["use_restricted_actions"], 0)
            self.assertEqual(report["gameplay_readiness"], "not_verified")
            self.assertEqual(report["milestones"], milestones)
            self.assertEqual(milestones["rom_integration_boots"]["status"], "partially_verified")
            self.assertEqual(milestones["input_controls_function"]["status"], "partially_verified")
            self.assertEqual(milestones["party_creation_works"]["status"], "not_verified")
            self.assertEqual(milestones["playable_scene_reached"]["status"], "not_verified")
            self.assertEqual(milestones["native_state_serialized"]["status"], "passed")
            self.assertEqual(milestones["state_restored_after_reset"]["status"], "passed")
            self.assertEqual(milestones["gameplay_resumes_correctly"]["status"], "not_verified")
            self.assertEqual(report["party_setup_trace_screenshots"], 1)
            self.assertTrue((output / "party_setup_trace.csv").is_file())
            self.assertEqual(len(report["party_setup_contact_sheets"]), 1)
            self.assertFalse(report["party_state_machine"]["party_setup_started"])
            self.assertIn("No menu action sent", report["party_state_machine"]["failure"])
            self.assertTrue((output / "party_state_machine_trace.json").is_file())
            self.assertTrue((output / "runtime_progress.json").is_file())

            failed_env = FakeEnv(fail_after=2)
            fake_retro.make = lambda **_kwargs: failed_env
            failed_output = project / "failed_run"
            failed_output.mkdir()
            with patch("scripts.gba_feasibility_probe.PROJECT_ROOT", project), patch.dict(sys.modules, {"stable_retro": fake_retro}):
                with self.assertRaisesRegex(RuntimeError, "synthetic input failure"):
                    runtime_smoke(failed_output, automate_party_setup=True)
            progress = json.loads((failed_output / "runtime_progress.json").read_text(encoding="utf-8"))
            self.assertIn("synthetic input failure", progress["failure"])
            self.assertTrue((failed_output / "runtime_progress.json").is_file())

    def test_party_trace_saves_input_frame_and_changed_pixel_fraction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            output = Path(temp)
            before = np.zeros((8, 8, 3), dtype=np.uint8)
            after = before.copy()
            after[0, 0] = 255
            rows = []
            _save_trace_frame(output, rows, 0, "A", "released", after, before)
            self.assertEqual(rows[0]["changed_pixel_fraction"], "0.015625")
            self.assertTrue((output / rows[0]["screenshot"]).is_file())
            _save_trace_frame(output, rows, 1, "DOWN", "released", after, after)
            self.assertEqual(rows[0]["screen_id"], rows[1]["screen_id"])
            trace = (output / "party_setup_trace.csv").read_text(encoding="utf-8")
            self.assertIn("A,released,0.015625", trace)
            summary = _summarize_party_trace(output, rows)
            self.assertEqual(summary["ordered_action_screens"], 2)
            self.assertEqual(len(summary["consecutive_exact_duplicate_screens"]), 1)
            self.assertTrue((output / summary["contact_sheets"][0]).is_file())
            self.assertFalse(summary["screen_semantics_classified"])

    def test_runtime_milestones_begin_unverified(self):
        milestones = _initial_milestones()
        self.assertEqual(milestones["rom_integration_boots"]["status"], "not_run")
        self.assertEqual(milestones["party_creation_works"]["status"], "not_verified")
        self.assertEqual(milestones["playable_scene_reached"]["status"], "not_verified")
        self.assertEqual(milestones["gameplay_resumes_correctly"]["status"], "not_verified")

    def test_macro_loop_guard_stops_only_on_repeated_exact_pair_frame(self):
        seen = {}
        frame_a = np.zeros((2, 3, 3), dtype=np.uint8)
        frame_b = np.ones((2, 3, 3), dtype=np.uint8)
        self.assertEqual(_repeated_pair_boundary_frame(frame_a, seen, 1), (False, ""))
        self.assertEqual(_repeated_pair_boundary_frame(frame_b, seen, 2), (False, ""))
        repeated, reason = _repeated_pair_boundary_frame(frame_a.copy(), seen, 3)
        self.assertTrue(repeated)
        self.assertIn("pair 3", reason)
        self.assertIn("pair 1", reason)

    def test_top_level_runtime_milestones_match_nested_runtime_results(self):
        milestones = _initial_milestones()
        milestones["native_state_serialized"] = {"status": "passed", "evidence": "synthetic"}
        runtime = {"status": "completed", "milestones": milestones}
        summary = {"status": "failed", "milestones": _initial_milestones()}
        _merge_runtime_results(summary, runtime)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["milestones"], runtime["milestones"])
        self.assertEqual(summary["gameplay_readiness"], "not_verified")
        self.assertEqual(runtime["gameplay_readiness"], "not_verified")

    def test_verified_playability_is_synchronized_to_nested_runtime(self):
        milestones = _initial_milestones()
        for key in (
            "four_characters_confirmed", "playable_field_reached",
            "native_checkpoint_restored", "field_movement_after_restore",
        ):
            milestones[key] = {"status": "passed", "evidence": "synthetic"}
        runtime = {"status": "completed", "milestones": milestones}
        summary = {}
        _merge_runtime_results(summary, runtime)
        self.assertEqual(summary["gameplay_readiness"], "verified")
        self.assertEqual(summary["runtime"]["gameplay_readiness"], "verified")

    def test_saved_menu_route_is_bounded_and_reaches_party_editor(self):
        self.assertEqual(len(BOOT_ROUTE), 5)
        self.assertEqual(BOOT_ROUTE[-1].expected, Screen.PARTY_SETUP)
        self.assertEqual([a.button for a in BOOT_ROUTE], ["START", "START", "A", "START", "A"])
        self.assertEqual(BOOT_ROUTE[-1].stage, "start-new-game-and-open-party")

    def test_unrecognized_menu_state_stops_without_guessing_next_input(self):
        observed = iter([Screen.BOOT, Screen.BOOT, *([Screen.UNKNOWN] * 21)])
        calls = []
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        with patch("scripts.gba_feasibility_probe.classify_screen", side_effect=lambda _frame: next(observed)):
            result_frame, result = _state_machine_setup(
                object(), {"START": 0, "A": 1, "SELECT": 2, "DOWN": 3},
                lambda button: calls.append(button) or frame,
                lambda button: calls.append(button) or frame,
                lambda _frames: frame,
                frame,
                Path(__file__).resolve().parent,
            )
        self.assertIs(result_frame, frame)
        self.assertEqual(calls, ["START"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("observed unknown", result["failure"])

    def test_custom_party_profiles_and_keyboard_navigation_are_exact(self):
        self.assertEqual([(m.display_name, m.job) for m in GBA_CUSTOM_PARTY], [
            ("Sif", "Warrior"), ("Myth", "Thief"),
            ("Milo", "Monk"), ("Seed", "Red Mage"),
        ])
        self.assertEqual([(m.display_name, m.job) for m in NES_SECONDARY_PARTY], [
            ("Sif", "Warrior"), ("Must", "White Mage"),
            ("Lock", "Black Mage"), ("Milo", "Monk"),
        ])
        self.assertEqual(name_key_position("S"), (1, 8))
        self.assertEqual(name_key_position("u"), (5, 0))
        self.assertEqual(class_navigation("Red Mage"), ("RIGHT", "DOWN"))
        with self.assertRaises(ValueError):
            name_key_position("!")

    def test_real_keyboard_cursor_anchor_tolerates_phase_drift_and_rejects_wrong_key(self):
        root = Path(__file__).resolve().parents[1]
        run = root / "videos/gba_feasibility/20261027_221444_366119Z"
        live_cursor = run / "party_trace_035_down_released_short.png"
        if not live_cursor.is_file():
            self.skipTest("local WSL keyboard trace is not present")
        frame = np.asarray(Image.open(live_cursor).convert("RGB"))
        self.assertTrue(name_key_cursor_matches(frame, "i"))
        self.assertFalse(name_key_cursor_matches(frame, "M"))
        class_trace = root / "videos/gba_feasibility/20261027_221759_222385Z/party_trace_073_right_released_short.png"
        if not class_trace.is_file():
            self.skipTest("local WSL class-picker trace is not present")
        thief_frame = np.asarray(Image.open(class_trace).convert("RGB"))
        self.assertEqual(class_picker_choice(thief_frame), "Thief")
        seed_anchor_dir = root / "videos/gba_feasibility/name_keyboard_probe/20261027_222332_657033Z"
        for letter in ("e", "d"):
            seed_frame = np.asarray(Image.open(seed_anchor_dir / f"key_target_{ord(letter):02x}.png").convert("RGB"))
            self.assertTrue(name_key_cursor_matches(seed_frame, letter))

    def test_gba_evidence_packager_requires_verified_milestones_and_sanitizes_paths(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            run = root / "20261027_test"
            run.mkdir()
            output = root / "published"
            members = [
                {"canonical_name": "Sif", "display_name": "Sif", "display_name_matches_verified_key_sequence": True},
                {"canonical_name": "Myth", "display_name": "Myth", "display_name_matches_verified_key_sequence": True},
                {"canonical_name": "Milo", "display_name": "Milo", "display_name_matches_verified_key_sequence": True},
                {"canonical_name": "Seed", "display_name": "Seed", "display_name_matches_verified_key_sequence": True},
            ]
            passed = {name: {"status": "passed"} for name in (
                "four_characters_confirmed", "party_setup_completed", "playable_field_reached",
                "native_checkpoint_saved", "native_checkpoint_restored", "field_movement_after_restore",
            )}
            metrics = {"runtime": {
                "gameplay_readiness": "verified", "native_state_bytes": 123,
                "party_state_machine": {"classes": ["Warrior", "Thief", "Monk", "Red Mage"],
                    "entered_names": members, "movement_verification": {"directions": {key: {
                        "passed": True, "idle_changed_pixel_fraction": 0, "movement_changed_pixel_fraction": 0.2
                    } for key in ("UP", "DOWN", "LEFT", "RIGHT")}}},
                "milestones": passed,
            }}
            (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            for filename in ("gba_custom_party_roster.png", "gba_after_reload_resume.png"):
                Image.new("RGB", (240, 160), (30, 90, 160)).save(run / filename)
            packaged = package_run(run, output, ["independent-run"])
            public = json.loads((packaged / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(public["matching_clean_run_count"], 2)
            self.assertEqual(public["party"][3]["display_name"], "Seed")
            report = (packaged / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("independent-run", report)
            self.assertNotIn(str(root), report)
            self.assertTrue((packaged / "proof.png").is_file())
            with Image.open(packaged / "proof.png") as proof:
                self.assertEqual(proof.size, (500, 198))
            with self.assertRaises(FileExistsError):
                package_run(run, output)
            metrics["runtime"]["gameplay_readiness"] = "not_verified"
            (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unverified gameplay"):
                package_run(run, root / "rejected")

    def test_name_change_is_row_specific(self):
        before = np.zeros((160, 240, 3), dtype=np.uint8)
        after = before.copy()
        after[43:60, 50:112] = 255
        self.assertTrue(name_changed(before, after, 1))
        self.assertFalse(name_changed(before, after, 0))

    def test_classifier_accepts_known_full_anchor_and_rejects_unknown(self):
        from ff1_dawn_ai import gba_party_setup

        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        with patch.object(gba_party_setup, "_sha256", return_value="a1f00ccd4c00135c75646919e4144d4e4cb2618dd239bb834786e5500a274cfb"):
            self.assertEqual(classify_screen(frame), Screen.DOS_TITLE)
        with patch.object(gba_party_setup, "_sha256", return_value="unseen"):
            self.assertEqual(classify_screen(frame), Screen.UNKNOWN)

    def test_classifier_recognizes_party_confirmation_panel(self):
        from ff1_dawn_ai import gba_party_setup

        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        with patch.object(gba_party_setup, "_sha256", side_effect=["unknown", "7ac10055b41a6e6580bdf7e349ef09ed73bd3a872de12fd7c3a54b74bff9df89"]):
            self.assertEqual(classify_screen(frame), Screen.PARTY_CONFIRM)

    def test_field_map_cue_requires_reload_movement_beyond_idle_variation(self):
        from ff1_dawn_ai import gba_party_setup

        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        frame[:] = (50, 160, 70)
        frame[100:160, 200:240] = (20, 50, 180)
        with patch.object(gba_party_setup, "_sha256", side_effect=lambda _img: "unknown"):
            self.assertTrue(looks_like_cornelia_field(frame))
        idle_a = frame.copy()
        idle_b = frame.copy()
        idle_b[10:12, 10:12] = (200, 10, 10)
        moved = idle_b.copy()
        moved[45:85, 40:95] = (200, 10, 10)
        evidence = meaningful_field_movement(idle_a, idle_b, moved)
        self.assertTrue(evidence["passed"])
        self.assertGreater(evidence["movement_changed_pixel_fraction"], evidence["idle_changed_pixel_fraction"])

    def test_reads_png_chunk_inventory_without_decoding_payloads(self):
        sample = (
            b"\x89PNG\r\n\x1a\n"
            + make_chunk(b"IHDR", b"header")
            + make_chunk(b"gbAs", b"state payload")
            + make_chunk(b"IEND", b"")
        )
        self.assertEqual(png_chunks(sample), [("IHDR", 6), ("gbAs", 13), ("IEND", 0)])

    def test_rejects_bad_crc_and_trailing_data(self):
        valid = b"\x89PNG\r\n\x1a\n" + make_chunk(b"IEND", b"")
        corrupted = valid[:-1] + bytes([valid[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, "bad CRC"):
            png_chunks(corrupted)
        with self.assertRaisesRegex(ValueError, "after PNG IEND"):
            png_chunks(valid + b"extra")

    def test_inspect_file_reports_size_and_chunk_metadata(self):
        sample = b"\x89PNG\r\n\x1a\n" + make_chunk(b"gbAs", b"x") + make_chunk(b"IEND", b"")
        path = Path(__file__).resolve().parent / f".fixture_{uuid.uuid4().hex}.ss0"
        try:
            path.write_bytes(sample)
            result = inspect_file(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(result["is_png_container"])
        self.assertEqual(result["size_bytes"], len(sample))
        self.assertEqual([c["type"] for c in result["png_chunks"]], ["gbAs", "IEND"])


if __name__ == "__main__":
    unittest.main()
