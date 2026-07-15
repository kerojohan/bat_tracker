#!/usr/bin/env python3
"""Benchmark all caves and require byte-identical v1.1.31 tracks.csv files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "out_v1_1_31_phase1_20260715/baseline"
CAVES = (
    "cova2",
    "rabella",
    "foric",
    "bora_tuna",
    "bofia_sant_jaume",
    "senioles",
    "crespia",
    "carradon",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _track_counts(path: Path) -> tuple[int, int]:
    ids = set()
    points = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ids.add(int(row["track_id"]))
            points += 1
    return len(ids), points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "out_v1_1_31_optimized_parity_20260715",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Directory containing the reference v1.1.31 output for each cave",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.out3_clean.yaml")
    parser.add_argument("--report", type=Path, default=ROOT / "benchmarks/v1_1_31_exact_parity.json")
    parser.add_argument("--caves", nargs="+", choices=CAVES)
    args = parser.parse_args()
    selected = args.caves or list(CAVES)
    missing = [cave for cave in selected if not (args.reference_root / cave / "tracks.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing v1.1.31 reference tracks in {args.reference_root}: {', '.join(missing)}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cave in selected:
        reference_dir = args.reference_root / cave
        reference_meta = json.loads((reference_dir / "meta.json").read_text(encoding="utf-8"))
        input_video = Path(reference_meta["video"]["input_path"])
        output_dir = args.output_root / cave
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark output: {output_dir}")
        log_path = args.output_root / f"{cave}.log"
        command = [
            sys.executable,
            "-m",
            "bat_tracker",
            "--input",
            str(input_video),
            "--output",
            str(output_dir),
            "--config",
            str(args.config),
        ]
        print(f"Benchmark {cave}...", flush=True)
        started = perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        wall_sec = perf_counter() - started
        candidate = output_dir / "tracks.csv"
        reference = reference_dir / "tracks.csv"
        reference_tracks, reference_points = _track_counts(reference)
        candidate_tracks, candidate_points = _track_counts(candidate)
        exact = candidate.read_bytes() == reference.read_bytes()
        baseline_sec = float(reference_meta["performance"]["pipeline_total_wall_sec"])
        row = {
            "cave": cave,
            "exact_tracks_csv": exact,
            "reference_sha256": _sha(reference),
            "candidate_sha256": _sha(candidate),
            "reference_tracks": reference_tracks,
            "candidate_tracks": candidate_tracks,
            "reference_points": reference_points,
            "candidate_points": candidate_points,
            "baseline_wall_sec": baseline_sec,
            "optimized_wall_sec": wall_sec,
            "speedup": baseline_sec / wall_sec,
        }
        rows.append(row)
        print(
            f"  exact={exact} tracks={candidate_tracks} "
            f"wall={wall_sec:.2f}s speedup={row['speedup']:.2f}x",
            flush=True,
        )
        if not exact:
            raise RuntimeError(f"Parity failure in {cave}: {candidate} != {reference}")

    baseline_total = sum(row["baseline_wall_sec"] for row in rows)
    optimized_total = sum(row["optimized_wall_sec"] for row in rows)
    report = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_commit": "d208d80e4603f523776bcb8d06a861ac73c78b80",
        "reference_name": "v1.1.31-out3-clean",
        "config": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "contract": "tracks.csv byte-for-byte equality for every cave",
        "all_exact": all(row["exact_tracks_csv"] for row in rows),
        "reference_tracks_total": sum(row["reference_tracks"] for row in rows),
        "candidate_tracks_total": sum(row["candidate_tracks"] for row in rows),
        "reference_points_total": sum(row["reference_points"] for row in rows),
        "candidate_points_total": sum(row["candidate_points"] for row in rows),
        "baseline_wall_sec_total": baseline_total,
        "optimized_wall_sec_total": optimized_total,
        "overall_speedup": baseline_total / optimized_total,
        "caves": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Exact parity: {report['candidate_tracks_total']} tracks; "
        f"{baseline_total:.1f}s -> {optimized_total:.1f}s ({report['overall_speedup']:.2f}x)."
    )


if __name__ == "__main__":
    main()
