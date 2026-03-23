from __future__ import annotations

import argparse
import calendar
import datetime as dt
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MonthlyWindow:
    start_date: dt.date
    end_date: dt.date

    @property
    def reviewed_bundle_name(self) -> str:
        return (
            "curtailment_opportunity_live_britned_reviewed_"
            f"{self.start_date.isoformat()}_{self.end_date.isoformat()}"
        )

    @property
    def truth_dir_name(self) -> str:
        return f"bmu_truth_history_{self.start_date.isoformat()}_{self.end_date.isoformat()}"

    @property
    def fleet_dir_name(self) -> str:
        return f"bmu_fleet_history_{self.start_date.isoformat()}_{self.end_date.isoformat()}"


def parse_month(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(f"{value}-01")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid month '{value}', expected YYYY-MM") from exc


def iter_months(start_month: dt.date, end_month: dt.date) -> list[MonthlyWindow]:
    windows: list[MonthlyWindow] = []
    current = start_month
    while current <= end_month:
        last_day = calendar.monthrange(current.year, current.month)[1]
        windows.append(
            MonthlyWindow(
                start_date=dt.date(current.year, current.month, 1),
                end_date=dt.date(current.year, current.month, last_day),
            )
        )
        if current.month == 12:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 1, 1)
    return windows


def _london_fallback_day(year: int) -> dt.date:
    last_day = dt.date(year, 10, 31)
    return last_day - dt.timedelta(days=(last_day.weekday() + 1) % 7)


def iter_reviewed_bundle_windows(
    month_windows: Iterable[MonthlyWindow],
    *,
    max_window_days: int = 7,
) -> list[MonthlyWindow]:
    if max_window_days <= 0:
        raise ValueError("max_window_days must be positive")

    reviewed_windows: list[MonthlyWindow] = []
    for month_window in month_windows:
        current_start = month_window.start_date
        while current_start <= month_window.end_date:
            current_end = min(
                month_window.end_date,
                current_start + dt.timedelta(days=max_window_days - 1),
            )
            fallback_day = (
                _london_fallback_day(current_start.year)
                if current_start.month == 10
                else None
            )
            if fallback_day is not None and current_start <= fallback_day <= current_end:
                if current_start < fallback_day:
                    reviewed_windows.append(
                        MonthlyWindow(
                            start_date=current_start,
                            end_date=fallback_day - dt.timedelta(days=1),
                        )
                    )
                split_day = fallback_day
                while split_day <= current_end:
                    reviewed_windows.append(
                        MonthlyWindow(
                            start_date=split_day,
                            end_date=split_day,
                        )
                    )
                    split_day += dt.timedelta(days=1)
            else:
                reviewed_windows.append(MonthlyWindow(start_date=current_start, end_date=current_end))
            current_start = current_end + dt.timedelta(days=1)
    return reviewed_windows


def run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, check=False)


def output_complete(output_dir: Path, sentinel_name: str) -> bool:
    return output_dir.exists() and (output_dir / sentinel_name).exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="Repo root containing inline_arbitrage_live.py")
    parser.add_argument("--run-root", required=True, help="Root output directory for monthly backfill")
    parser.add_argument("--start-month", type=parse_month, required=True, help="First month to materialize (YYYY-MM)")
    parser.add_argument("--end-month", type=parse_month, required=True, help="Last month to materialize (YYYY-MM)")
    parser.add_argument(
        "--materialize-reviewed-bundles",
        action="store_true",
        help="Materialize full-month reviewed opportunity bundles",
    )
    parser.add_argument(
        "--materialize-bmu-truth",
        action="store_true",
        help="Materialize full-month wind truth outputs with physical/availability/bid layers",
    )
    parser.add_argument(
        "--materialize-bmu-fleet",
        action="store_true",
        help="Materialize full-month all-fuel BMU fleet outputs for displacement work",
    )
    parser.add_argument(
        "--run-reviewed-bundle-batch-eval",
        action="store_true",
        help="Run the existing reviewed-bundle batch evaluation over the new bundle root",
    )
    parser.add_argument(
        "--opportunity-truth-profile",
        default="proxy",
        choices=("proxy", "research", "precision", "all"),
        help="Truth profile for reviewed opportunity bundles",
    )
    parser.add_argument(
        "--truth-profile",
        default="all",
        choices=("all", "precision", "research"),
        help="Truth profile for BMU curtailment truth outputs",
    )
    parser.add_argument(
        "--backtest-model-key",
        default="opportunity_potential_ratio_v2",
        help="Model key for batch evaluation",
    )
    parser.add_argument(
        "--baseline-model-key",
        default="opportunity_potential_ratio_v2",
        help="Baseline model key for batch evaluation",
    )
    parser.add_argument(
        "--backtest-horizons",
        default="1,6,24,168",
        help="Comma-separated forecast horizons for batch evaluation",
    )
    parser.add_argument(
        "--reviewed-window-days",
        type=int,
        default=7,
        help="Maximum days per reviewed opportunity bundle window (GB Elexon MID currently supports up to 7)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip month outputs that already exist",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the monthly plan without materializing anything",
    )
    args = parser.parse_args()

    if args.end_month < args.start_month:
        raise SystemExit("--end-month must be on or after --start-month")
    if not any(
        [
            args.materialize_reviewed_bundles,
            args.materialize_bmu_truth,
            args.materialize_bmu_fleet,
            args.run_reviewed_bundle_batch_eval,
            args.plan_only,
        ]
    ):
        raise SystemExit("select at least one action")

    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    bundle_root = run_root / "bundles"
    truth_root = run_root / "truth_months"
    fleet_root = run_root / "fleet_months"
    month_windows = iter_months(args.start_month, args.end_month)
    reviewed_windows = iter_reviewed_bundle_windows(month_windows, max_window_days=args.reviewed_window_days)

    print(
        f"[plan] months={len(month_windows)} reviewed_windows={len(reviewed_windows)} "
        f"reviewed_window_days={args.reviewed_window_days} run_root={run_root}"
    )
    for window in month_windows:
        print(f"{window.start_date.isoformat()}..{window.end_date.isoformat()}")
    if args.plan_only:
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    truth_root.mkdir(parents=True, exist_ok=True)
    fleet_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    executed = 0
    skipped = 0

    if args.materialize_reviewed_bundles:
        for window in reviewed_windows:
            output_dir = bundle_root / window.reviewed_bundle_name
            if args.skip_existing and output_complete(output_dir, "fact_curtailment_opportunity_hourly.csv"):
                skipped += 1
                print(f"[skip] reviewed_bundle {window.reviewed_bundle_name}")
            else:
                command = [
                    sys.executable,
                    "inline_arbitrage_live.py",
                    "--materialize-curtailment-opportunity-history",
                    "--opportunity-start",
                    window.start_date.isoformat(),
                    "--opportunity-end",
                    window.end_date.isoformat(),
                    "--opportunity-output-dir",
                    str(output_dir),
                    "--opportunity-truth-profile",
                    args.opportunity_truth_profile,
                ]
                print(f"[run] reviewed_bundle {window.reviewed_bundle_name}")
                result = run_command(command, cwd=repo_root)
                executed += 1
                if result.returncode != 0:
                    failures += 1
                    print(result.stdout)
                    print(result.stderr, file=sys.stderr)

    for window in month_windows:
        if args.materialize_bmu_truth:
            output_dir = truth_root / window.truth_dir_name
            if args.skip_existing and output_complete(output_dir, "fact_bmu_curtailment_truth_half_hourly.csv"):
                skipped += 1
                print(f"[skip] bmu_truth {window.truth_dir_name}")
            else:
                command = [
                    sys.executable,
                    "inline_arbitrage_live.py",
                    "--materialize-bmu-curtailment-truth",
                    "--truth-start",
                    window.start_date.isoformat(),
                    "--truth-end",
                    window.end_date.isoformat(),
                    "--truth-output-dir",
                    str(output_dir),
                    "--truth-profile",
                    args.truth_profile,
                ]
                print(f"[run] bmu_truth {window.truth_dir_name}")
                result = run_command(command, cwd=repo_root)
                executed += 1
                if result.returncode != 0:
                    failures += 1
                    print(result.stdout)
                    print(result.stderr, file=sys.stderr)

        if args.materialize_bmu_fleet:
            output_dir = fleet_root / window.fleet_dir_name
            if args.skip_existing and output_complete(output_dir, "fact_bmu_generation_half_hourly.csv"):
                skipped += 1
                print(f"[skip] bmu_fleet {window.fleet_dir_name}")
            else:
                command = [
                    sys.executable,
                    "bmu_fleet_history.py",
                    "--start",
                    window.start_date.isoformat(),
                    "--end",
                    window.end_date.isoformat(),
                    "--output-dir",
                    str(output_dir),
                ]
                print(f"[run] bmu_fleet {window.fleet_dir_name}")
                result = run_command(command, cwd=repo_root)
                executed += 1
                if result.returncode != 0:
                    failures += 1
                    print(result.stdout)
                    print(result.stderr, file=sys.stderr)

    if args.run_reviewed_bundle_batch_eval:
        batch_output_dir = run_root / "model_readiness_reviewed_bundle_batch"
        command = [
            sys.executable,
            "inline_arbitrage_live.py",
            "--materialize-reviewed-bundle-batch-eval",
            "--reviewed-bundle-batch-root",
            str(bundle_root),
            "--reviewed-bundle-batch-output-dir",
            str(batch_output_dir),
            "--backtest-model-key",
            args.backtest_model_key,
            "--baseline-model-key",
            args.baseline_model_key,
            "--backtest-horizons",
            args.backtest_horizons,
        ]
        print(f"[run] reviewed_bundle_batch_eval output={batch_output_dir}")
        result = run_command(command, cwd=repo_root)
        executed += 1
        if result.returncode != 0:
            failures += 1
            print(result.stdout)
            print(result.stderr, file=sys.stderr)

    print(
        f"[summary] executed={executed} skipped={skipped} failures={failures} "
        f"run_root={run_root}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
