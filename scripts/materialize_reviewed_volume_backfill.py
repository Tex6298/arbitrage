from __future__ import annotations

import argparse
import calendar
import datetime as dt
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BUNDLE_PATTERN = "curtailment_opportunity_live_britned_reviewed*"


@dataclass(frozen=True)
class MonthlyWindowSpec:
    start_date: dt.date
    end_date: dt.date

    @property
    def bundle_name(self) -> str:
        return (
            "curtailment_opportunity_live_britned_reviewed_"
            f"{self.start_date.isoformat()}_{self.end_date.isoformat()}"
        )


def parse_month(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(f"{value}-01")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid month '{value}', expected YYYY-MM") from exc


def iter_month_starts(start_month: dt.date, end_month: dt.date) -> Iterable[dt.date]:
    current = dt.date(start_month.year, start_month.month, 1)
    limit = dt.date(end_month.year, end_month.month, 1)
    while current <= limit:
        yield current
        if current.month == 12:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 1, 1)


def build_monthly_windows(
    start_month: dt.date,
    end_month: dt.date,
    *,
    anchor_day: int = 15,
    window_days: int = 3,
) -> list[MonthlyWindowSpec]:
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if anchor_day <= 0:
        raise ValueError("anchor_day must be positive")

    windows: list[MonthlyWindowSpec] = []
    for month_start in iter_month_starts(start_month, end_month):
        month_last_day = calendar.monthrange(month_start.year, month_start.month)[1]
        latest_valid_start = max(1, month_last_day - window_days + 1)
        start_day = min(anchor_day, latest_valid_start)
        start_date = dt.date(month_start.year, month_start.month, start_day)
        end_date = start_date + dt.timedelta(days=window_days - 1)
        windows.append(MonthlyWindowSpec(start_date=start_date, end_date=end_date))
    return windows


def copy_existing_bundles(source_root: Path, destination_root: Path, *, pattern: str = DEFAULT_BUNDLE_PATTERN) -> list[Path]:
    copied_paths: list[Path] = []
    destination_root.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(path for path in source_root.glob(pattern) if path.is_dir()):
        destination_path = destination_root / source_path.name
        if destination_path.exists():
            continue
        shutil.copytree(source_path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


def materialize_window(repo_root: Path, output_dir: Path, window: MonthlyWindowSpec) -> subprocess.CompletedProcess[str]:
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
    ]
    return subprocess.run(command, cwd=repo_root, text=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="Repo root containing inline_arbitrage_live.py")
    parser.add_argument("--bundle-root", required=True, help="Destination directory for materialized reviewed bundles")
    parser.add_argument("--start-month", type=parse_month, required=True, help="First month to materialize (YYYY-MM)")
    parser.add_argument("--end-month", type=parse_month, required=True, help="Last month to materialize (YYYY-MM)")
    parser.add_argument("--anchor-day", type=int, default=15, help="Anchor day within each month")
    parser.add_argument("--window-days", type=int, default=3, help="Window length in days per month")
    parser.add_argument(
        "--copy-existing-bundles",
        action="store_true",
        help="Copy current root reviewed bundles into bundle-root before materializing new monthly windows",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip monthly windows whose output directory already exists",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    bundle_root = Path(args.bundle_root).resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)

    if args.end_month < args.start_month:
        raise SystemExit("--end-month must be on or after --start-month")

    if args.copy_existing_bundles:
        copied = copy_existing_bundles(repo_root, bundle_root)
        print(f"[copy] copied {len(copied)} existing reviewed bundles into {bundle_root}")

    windows = build_monthly_windows(
        args.start_month,
        args.end_month,
        anchor_day=args.anchor_day,
        window_days=args.window_days,
    )

    failures = 0
    materialized = 0
    skipped = 0
    for window in windows:
        output_dir = bundle_root / window.bundle_name
        if args.skip_existing and output_dir.exists():
            skipped += 1
            print(f"[skip] {window.bundle_name}")
            continue
        print(f"[run] {window.bundle_name}")
        result = materialize_window(repo_root, output_dir, window)
        if result.returncode != 0:
            failures += 1
            print(f"[fail] {window.bundle_name}")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        else:
            materialized += 1

    print(
        f"[summary] copied_existing={args.copy_existing_bundles} materialized={materialized} "
        f"skipped={skipped} failures={failures} bundle_root={bundle_root}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
