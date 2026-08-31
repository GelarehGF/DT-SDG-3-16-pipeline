"""Tests for the one-command runner; stage stubs test orchestration, not science."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workflow_main", ROOT / "main.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source with spaces"
        self.source.mkdir()
        self.output = self.root / "results with spaces"
        self.argv = ["--source", str(self.source), "--output", str(self.output)]

    def tearDown(self):
        self.temp.cleanup()

    def test_all_five_stages_use_same_interpreter_and_connected_paths(self):
        args = runner.parse_args(self.argv)
        stages = runner.build_stages(args)
        self.assertEqual([stage.key for stage in stages], ["master", "network", "analysis_figures", "readable_figures", "network_figures"])
        self.assertTrue(all(stage.command[:2] == [sys.executable, "-u"] for stage in stages))
        self.assertIn(str(self.source), stages[0].command)
        self.assertIn(str(self.output / "master"), stages[1].command)
        self.assertIn(str(self.output / "network"), stages[-1].command)
        self.assertIn("--fail-on-workbook-error", stages[0].command)
        self.assertTrue(any(path.suffix == ".xlsx" for path in stages[0].required_outputs))

    def test_custom_parameters_and_fonts_propagate(self):
        args = runner.parse_args(self.argv + ["--seed", "7", "--primary-k", "4", "--sensitivity-k", "2", "4", "6", "--regular-font", str(self.root / "custom.ttf")])
        stages = runner.build_stages(args)
        for stage in (stages[1], stages[4]):
            self.assertEqual(stage.command[stage.command.index("--seed") + 1], "7")
        for stage in stages[3:]:
            self.assertIn(str(self.root / "custom.ttf"), stage.command)
        self.assertEqual(stages[1].command[stages[1].command.index("--primary-k") + 1], "4")

    def test_skip_workbook_does_not_skip_other_stages(self):
        stages = runner.build_stages(runner.parse_args(self.argv + ["--skip-workbook"]))
        self.assertEqual(len(stages), 5)
        self.assertIn("--skip-workbook", stages[0].command)
        self.assertFalse(any(path.suffix == ".xlsx" for path in stages[0].required_outputs))

    def test_dry_run_does_not_write_or_launch_children(self):
        with patch.object(runner, "run_stage") as execute, contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(runner.main(self.argv + ["--dry-run"]), 0)
        self.assertFalse(self.output.exists())
        execute.assert_not_called()
        self.assertIn("[5/5]", stdout.getvalue())

    def test_default_output_is_timestamped(self):
        args = runner.parse_args(["--source", str(self.source)])
        self.assertEqual(args.output.parent, ROOT / "outputs")
        self.assertTrue(args.output.name.startswith("run-"))

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        existing = self.output / "human_review.txt"
        existing.write_text("Keep completed review")
        with contextlib.redirect_stderr(io.StringIO()), patch.object(runner, "run_stage") as execute:
            self.assertEqual(runner.main(self.argv), 2)
        execute.assert_not_called()
        self.assertEqual(existing.read_text(), "Keep completed review")

    def test_nested_output_is_rejected(self):
        args = runner.parse_args(["--source", str(self.source), "--output", str(self.source / "results")])
        with self.assertRaisesRegex(ValueError, "non-nested"):
            runner.validate_paths(args)

    def test_missing_dependencies_fail_before_writes(self):
        with patch.object(runner.util, "find_spec", return_value=None), contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(runner.main(self.argv), 2)
        self.assertIn("Missing dependencies", stderr.getvalue())
        self.assertFalse(self.output.exists())

    @staticmethod
    def successful_stub(stage, log_path, environment):
        if environment["MPLBACKEND"] != "Agg":
            raise AssertionError("Headless backend not selected")
        log_path.write_text(stage.key)
        for path in stage.required_outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic stage artifact")
        return 0

    def test_success_writes_complete_manifest_and_five_logs(self):
        with patch.object(runner, "dependency_versions", return_value={}), patch.object(runner, "run_stage", side_effect=self.successful_stub) as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(self.argv), 0)
        self.assertEqual(execute.call_count, 5)
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(all(stage["status"] == "complete" for stage in manifest["stages"]))
        self.assertEqual(len(list((self.output / "logs").glob("*.log"))), 5)
        self.assertIn(str(ROOT / "main.py"), manifest["script_and_codebook_sha256"])

    def test_child_failure_stops_later_stages(self):
        def fail_second(stage, log_path, environment):
            if stage.key == "network":
                log_path.write_text("Synthetic failure")
                return 7
            return self.successful_stub(stage, log_path, environment)
        with patch.object(runner, "dependency_versions", return_value={}), patch.object(runner, "run_stage", side_effect=fail_second) as execute, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(self.argv), 1)
        self.assertEqual(execute.call_count, 2)
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stages"][0]["status"], "complete")
        self.assertEqual(manifest["stages"][1]["status"], "failed")
        self.assertEqual(manifest["stages"][1]["exit_code"], 7)
        self.assertTrue((self.output / "master/master_papers.csv").exists())

    def test_missing_output_is_not_reported_as_success(self):
        with patch.object(runner, "dependency_versions", return_value={}), patch.object(runner, "run_stage", return_value=0) as execute, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(self.argv), 1)
        self.assertEqual(execute.call_count, 1)
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertIn("required outputs", manifest["stages"][0]["error"])

    def test_interrupt_is_recorded_and_stops_the_run(self):
        with patch.object(runner, "dependency_versions", return_value={}), patch.object(runner, "run_stage", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(self.argv), 130)
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")

    def test_real_child_output_is_streamed_and_logged(self):
        stage = runner.Stage("probe", "Probe", [sys.executable, "-u", "-c", "print('synthetic child output')"], ())
        log = self.root / "stage.log"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = runner.run_stage(stage, log, os.environ.copy())
        self.assertEqual(code, 0)
        self.assertIn("synthetic child output", stdout.getvalue())
        self.assertEqual(log.read_text(), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
