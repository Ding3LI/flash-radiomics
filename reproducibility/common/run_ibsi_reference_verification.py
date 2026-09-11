#!/usr/bin/env python3
"""Run the project cross-tool IBSI-1 and IBSI-2 verification suites."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = WORKSPACE_ROOT
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from reproducibility.common.ibsi2_cross_tool_validation import (
    ALLOWED_FLASH_BACKENDS,
    ALLOWED_TOOLS,
    DEFAULT_FLASH_BACKENDS,
    DEFAULT_TOOLS,
    _parse_csv_choices,
    run_cross_tool_ibsi2_validation,
)
from reproducibility.common.official_benchmark_common import write_json


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return WORKSPACE_ROOT / "reproducibility" / "common" / "exp" / f"{timestamp}_ibsi_reference_verification"


def _relative_repository_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _summary_path_from_stdout(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        if not line.startswith("summary_json="):
            continue
        reported = Path(line.split("=", 1)[1].strip())
        if reported.is_absolute():
            return reported
        candidate = WORKSPACE_ROOT / reported
        return candidate
    return None


def _ibsi1_jobs(
    *,
    tools: tuple[str, ...],
    flash_backends: tuple[str, ...],
    num_threads: int,
    repeats: int,
) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    if "flash" in tools:
        for backend in flash_backends:
            jobs.append(
                (
                    f"flash_{backend}",
                    [
                        sys.executable,
                        "reproducibility/flash/benchmark_flash_segment.py",
                        "--dataset-suite",
                        "ibsi1",
                        "--backend",
                        backend,
                        "--num-threads",
                        str(num_threads),
                        "--segment-class-workers",
                        "1",
                        "--repeats",
                        str(repeats),
                        "--skip-feature-timings",
                    ],
                )
            )
    if "mirp" in tools:
        jobs.append(
            (
                "mirp_cpu",
                [
                    sys.executable,
                    "reproducibility/mirp/benchmark_mirp_ibsi1_segment.py",
                    "--repeats",
                    str(repeats),
                ],
            )
        )
    if "pyradiomics" in tools:
        jobs.append(
            (
                "pyradiomics_cpu",
                [
                    sys.executable,
                    "reproducibility/pyradiomics/benchmark_pyradiomics_segment.py",
                    "--dataset-suite",
                    "ibsi1",
                    "--repeats",
                    str(repeats),
                ],
            )
        )
    return jobs


def _run_ibsi1_jobs(
    *,
    jobs: list[tuple[str, list[str]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    log_dir = output_dir / "ibsi1" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    for name, command in jobs:
        print(f"suite=ibsi1 job={name} status=running", flush=True)
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path = log_dir / f"{name}.stdout.log"
        stderr_path = log_dir / f"{name}.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        summary_path = _summary_path_from_stdout(completed.stdout)
        outcome = {
            "job": name,
            "command": ["python", *command[1:]],
            "return_code": completed.returncode,
            "summary_json": "" if summary_path is None else _relative_repository_path(summary_path),
            "summary_exists": bool(summary_path and summary_path.exists()),
            "stdout_log": _relative_repository_path(stdout_path),
            "stderr_log": _relative_repository_path(stderr_path),
        }
        outcomes.append(outcome)
        print(
            f"suite=ibsi1 job={name} status={'complete' if completed.returncode == 0 else 'failed'}",
            flush=True,
        )
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cross-tool IBSI-1 and IBSI-2 scalar reference verification."
    )
    parser.add_argument("--suite", choices=("both", "ibsi1", "ibsi2"), default="both")
    parser.add_argument("--tools", default=",".join(DEFAULT_TOOLS))
    parser.add_argument("--flash-backends", default=",".join(DEFAULT_FLASH_BACKENDS))
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tolerance-floor", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        tools = _parse_csv_choices(args.tools, ALLOWED_TOOLS, "--tools")
        flash_backends = _parse_csv_choices(
            args.flash_backends,
            ALLOWED_FLASH_BACKENDS,
            "--flash-backends",
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.tolerance_floor < 0:
        parser.error("--tolerance-floor must be non-negative")

    output_dir = args.out_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "suite": args.suite,
        "tools": list(tools),
        "flash_backends": list(flash_backends),
        "num_threads": args.num_threads,
        "repeats": args.repeats,
        "tolerance_floor": args.tolerance_floor,
        "ibsi1": None,
        "ibsi2": None,
    }
    has_errors = False

    if args.suite in {"both", "ibsi1"}:
        jobs = _ibsi1_jobs(
            tools=tools,
            flash_backends=flash_backends,
            num_threads=args.num_threads,
            repeats=args.repeats,
        )
        outcomes = _run_ibsi1_jobs(jobs=jobs, output_dir=output_dir)
        manifest["ibsi1"] = {"jobs": outcomes}
        has_errors = has_errors or any(
            outcome["return_code"] != 0 or not outcome["summary_exists"]
            for outcome in outcomes
        )

    if args.suite in {"both", "ibsi2"}:
        ibsi2_dir = output_dir / "ibsi2"
        _result_dir, summary = run_cross_tool_ibsi2_validation(
            tools=tools,
            flash_backends=flash_backends,
            num_threads=args.num_threads,
            tolerance_floor=args.tolerance_floor,
            output_dir=ibsi2_dir,
        )
        manifest["ibsi2"] = summary
        has_errors = has_errors or bool(summary["runtime_error_count"])

    manifest["completed_without_runtime_errors"] = not has_errors
    manifest_path = output_dir / "ibsi_reference_verification_manifest.json"
    write_json(manifest_path, manifest)
    print(f"out_dir={_relative_repository_path(output_dir)}")
    print(f"manifest_json={_relative_repository_path(manifest_path)}")
    print(f"completed_without_runtime_errors={not has_errors}")
    if has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
