#!/usr/bin/env python3
"""Run the complete revised DT–SDG 3/16 workflow with one command.

Example: python main.py --source /path/to/SDG3-16
The original notebook and manuscript text are not rerun or rewritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "technology_outcome_pipeline"


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    command: list[str]
    required_outputs: tuple[Path, ...]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "data/files", help="Read-only canonical corpus root")
    parser.add_argument("--output", type=Path, help="New or empty run folder; default: outputs/run-<UTC timestamp>")
    parser.add_argument("--codebook", type=Path, default=SCRIPTS / "codebook.json")
    parser.add_argument("--skip-workbook", action="store_true", help="Skip optional XLSX; CSV/JSON and every analysis/figure stage still run")
    parser.add_argument("--dry-run", action="store_true", help="Show all five commands without executing or creating files")
    parser.add_argument("--max-evidence-per-pair", type=int, default=3)
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--sensitivity-k", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--svd-components", type=int, default=200)
    parser.add_argument("--max-text-chars", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regular-font", type=Path)
    parser.add_argument("--bold-font", type=Path)
    parser.add_argument("--context-quantile", type=float, default=0.85)
    parser.add_argument("--cross-edges", type=int, default=150)
    args = parser.parse_args(argv)
    args.source = args.source.expanduser().resolve()
    args.codebook = args.codebook.expanduser().resolve()
    if args.output is None:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        args.output = ROOT / "outputs" / f"run-{suffix}"
    args.output = args.output.expanduser().resolve()
    for name in ("regular_font", "bold_font"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    positive = [args.max_evidence_per_pair, args.primary_k, *args.sensitivity_k,
                args.max_features, args.min_df, args.max_text_chars]
    if min(positive) < 1 or args.svd_components < 2:
        parser.error("Counts must be positive; --svd-components must be at least 2.")
    if not 0 <= args.min_similarity <= 1 or not 0 <= args.context_quantile <= 1:
        parser.error("Similarity cutoff and context quantile must be in [0, 1].")
    if args.cross_edges < 0 or not 0 <= args.seed < 2**32:
        parser.error("Cross-edge limit must be nonnegative; seed must be in [0, 2**32).")
    return args


def build_stages(args: argparse.Namespace) -> list[Stage]:
    master = args.output / "master"
    network = args.output / "network"
    analysis = args.output / "analysis_figures"
    figures = args.output / "manuscript_figures"

    def command(script: str, *values: object) -> list[str]:
        return [sys.executable, "-u", str(SCRIPTS / script), *map(str, values)]

    fonts = []
    for option in ("regular_font", "bold_font"):
        if getattr(args, option) is not None:
            fonts.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
    master_outputs = [master / name for name in (
        "master_papers.csv", "master_technology_outcome_relations.csv",
        "cross_sdg_relations.csv", "quality_report.json", "pipeline_manifest.json")]
    if not args.skip_workbook:
        master_outputs.append(master / "DT_SDG3_16_Technology_Outcome_Master.xlsx")
    return [
        Stage("master", "Extract technology–outcome master data", command(
            "pipeline.py", "--source", args.source, "--output", master,
            "--codebook", args.codebook, "--max-evidence-per-pair", args.max_evidence_per_pair,
            "--skip-workbook" if args.skip_workbook else "--fail-on-workbook-error"), tuple(master_outputs)),
        Stage("network", "Build paper similarity and knowledge networks", command(
            "build_knowledge_network.py", "--master-dir", master, "--output", network,
            "--codebook", args.codebook, "--primary-k", args.primary_k,
            "--sensitivity-k", *args.sensitivity_k, "--min-similarity", args.min_similarity,
            "--max-features", args.max_features, "--min-df", args.min_df,
            "--svd-components", args.svd_components, "--max-text-chars", args.max_text_chars,
            "--seed", args.seed), tuple(network / name for name in (
                "network_summary.json", "network_config.json", "paper_similarity_edges.csv",
                "paper_network_metrics.csv", "knowledge_network.graphml", "knowledge_network.gexf",
                "supporting_data/network_figure1_positions.csv"))),
        Stage("analysis_figures", "Generate analytical figures and supporting tables", command(
            "make_figures.py", "--input", master, "--output", analysis), tuple(analysis / name for name in (
                "figure_manifest.json", "supporting_data/figure1_technology_by_sdg.csv",
                "supporting_data/figure2_heatmap_sdg3.csv", "supporting_data/figure2_heatmap_sdg16.csv"))),
        Stage("readable_figures", "Generate readable manuscript charts", command(
            "make_manuscript_figures_v09.py", "--network-dir", network,
            "--figure-data-dir", analysis / "supporting_data", "--output", figures, *fonts),
            tuple(figures / name for name in (
                "figure3_similarity_distributions_readable.png", "figure4_technology_bars_readable.png",
                "figure5_heatmaps_readable.png", "figure6_aggregate_network_readable.png",
                "figure7_cross_sdg_readable.png", "figure8_evidence_profile_readable.png"))),
        Stage("network_figures", "Generate within-SDG, combined, full and community networks", command(
            "make_network_figures_v10.py", "--network-dir", network, "--output", figures,
            "--seed", args.seed, "--context-quantile", args.context_quantile,
            "--cross-edges", args.cross_edges, *fonts), tuple(figures / name for name in (
                "figure3_within_sdg_networks.png", "figure4_filtered_combined_network.png",
                "supplementary_figure_s1_full_network.png", "supplementary_figure_s2_communities.png",
                "network_figure_manifest.json"))),
    ]


def validate_paths(args: argparse.Namespace) -> None:
    if not args.source.is_dir():
        raise ValueError(f"Source folder not found: {args.source}. Supply --source /path/to/SDG3-16.")
    if not args.codebook.is_file():
        raise ValueError(f"Codebook not found: {args.codebook}")
    if args.source == args.output or args.source in args.output.parents or args.output in args.source.parents:
        raise ValueError("Source and output folders must be separate and non-nested.")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Output folder is not empty. Choose a new --output folder or omit --output for a fresh timestamped run.")
    for name in ("regular_font", "bold_font"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            raise ValueError(f"Font not found: {path}")
    for stage in build_stages(args):
        if not Path(stage.command[2]).is_file():
            raise ValueError(f"Required script missing: {stage.command[2]}. Download the complete repository, not main.py alone.")


def dependency_versions(skip_workbook: bool) -> dict[str, str]:
    packages = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
                "matplotlib": "matplotlib", "networkx": "networkx", "Pillow": "PIL"}
    if not skip_workbook:
        packages["openpyxl"] = "openpyxl"
    missing = [name for name, module in packages.items() if util.find_spec(module) is None]
    if missing:
        raise ValueError("Missing dependencies: " + ", ".join(missing) + ". Install requirements.txt using this Python environment.")
    return {name: metadata.version(name) for name in packages}


def run_stage(stage: Stage, log_path: Path, environment: dict[str, str]) -> int:
    """Stream child output to both the terminal and a per-stage log; never use a shell."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(stage.command, cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_paths(args)
        stages = build_stages(args)
        if args.dry_run:
            for number, stage in enumerate(stages, 1):
                display = subprocess.list2cmdline(stage.command) if os.name == "nt" else shlex.join(stage.command)
                print(f"[{number}/{len(stages)}] {stage.label}\n{display}\n")
            return 0
        versions = dependency_versions(args.skip_workbook)
    except (ValueError, OSError, metadata.PackageNotFoundError) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir()
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(args.output / ".cache/matplotlib")
    environment["XDG_CACHE_HOME"] = str(args.output / ".cache")
    environment["PYTHONUNBUFFERED"] = "1"
    fingerprints = [Path(__file__), args.codebook, SCRIPTS / "figure_fonts.py"]
    fingerprints.extend(Path(stage.command[2]) for stage in stages)
    manifest = {
        "status": "running", "started_at_utc": timestamp(),
        "python": sys.version, "python_executable": sys.executable,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dependency_versions": versions,
        "script_and_codebook_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in fingerprints},
        "stages": [],
        "privacy_note": "Local paths and evidence outputs are private; review before sharing.",
    }
    manifest_path = args.output / "run_manifest.json"

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    print(f"Run folder: {args.output}", flush=True)
    for number, stage in enumerate(stages, 1):
        log_path = logs / f"{number:02d}_{stage.key}.log"
        record = {"stage": stage.key, "command": stage.command, "log": str(log_path.relative_to(args.output)),
                  "status": "running", "started_at_utc": timestamp()}
        manifest["stages"].append(record)
        save_manifest()
        print(f"\n[{number}/{len(stages)}] {stage.label}", flush=True)
        try:
            code = run_stage(stage, log_path, environment)
            record["exit_code"] = code
            if code:
                raise RuntimeError(f"Stage exited with code {code}.")
            missing = [str(path.relative_to(args.output)) for path in stage.required_outputs if not path.is_file() or not path.stat().st_size]
            if missing:
                raise RuntimeError("Stage did not produce required outputs: " + ", ".join(missing))
            record.update(status="complete", finished_at_utc=timestamp(), exit_code=0)
        except (OSError, RuntimeError, KeyboardInterrupt) as error:
            interrupted = isinstance(error, KeyboardInterrupt)
            status = "interrupted" if interrupted else "failed"
            record.update(status=status, finished_at_utc=timestamp(), error=str(error))
            manifest.update(status=status, finished_at_utc=timestamp())
            save_manifest()
            print(f"\nStopped at {stage.key}. {error}\nLog: {log_path}\nCompleted outputs have been retained; later stages were not run.", file=sys.stderr)
            return 130 if interrupted else 1
        save_manifest()
    manifest.update(status="complete", finished_at_utc=timestamp())
    save_manifest()
    print(f"\nComplete. Master data, networks, charts and supplements: {args.output}\nRun record: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
