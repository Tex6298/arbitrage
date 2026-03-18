from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCOPE_PREFIXES = (
    "curtailment_opportunity",
    "model_readiness",
    "exploratory_cluster_map",
    "benchmark_window_scout",
)
REQUIRED_COLUMNS = (
    "path",
    "artifact_family",
    "artifact_kind",
    "action",
    "authority_state",
    "replacement_path",
    "archive_destination",
    "git_tracked_expected",
    "reason",
    "notes",
)
VALID_ACTIONS = ("keep", "archive", "delete")
VALID_SHOW = ("all",) + VALID_ACTIONS
VALID_FORMATS = ("table", "json")
ARCHIVE_PREFIX = Path("_local_archive/generated_outputs")


class ManifestValidationError(RuntimeError):
    """Raised when the cleanup manifest is incomplete or inconsistent."""


@dataclass(frozen=True)
class CleanupManifestRow:
    path: str
    artifact_family: str
    artifact_kind: str
    action: str
    authority_state: str
    replacement_path: str
    archive_destination: str
    git_tracked_expected: bool
    reason: str
    notes: str


@dataclass(frozen=True)
class CleanupReportRow:
    path: str
    artifact_family: str
    artifact_kind: str
    action: str
    authority_state: str
    replacement_path: str
    archive_destination: str
    git_tracked_expected: bool
    git_tracked_actual: bool
    git_tracked_mismatch: bool
    size_bytes: int
    reason: str
    notes: str


@dataclass(frozen=True)
class ActionSummary:
    action: str
    directory_count: int
    total_size_bytes: int
    tracked_count: int
    untracked_count: int
    mismatch_count: int


@dataclass(frozen=True)
class CleanupDryRunReport:
    repo_root: str
    manifest_path: str
    summaries: tuple[ActionSummary, ...]
    rows: tuple[CleanupReportRow, ...]


def is_scoped_generated_artifact(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in SCOPE_PREFIXES)


def parse_manifest_bool(value: str, *, column: str, row_path: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ManifestValidationError(
        f"{row_path}: {column} must be true or false; got {value!r}"
    )


def load_cleanup_manifest(manifest_path: Path) -> tuple[CleanupManifestRow, ...]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise ManifestValidationError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        rows: list[CleanupManifestRow] = []
        for raw_row in reader:
            path = (raw_row.get("path") or "").strip()
            if not path:
                raise ManifestValidationError("Manifest row is missing path")
            rows.append(
                CleanupManifestRow(
                    path=path,
                    artifact_family=(raw_row.get("artifact_family") or "").strip(),
                    artifact_kind=(raw_row.get("artifact_kind") or "").strip(),
                    action=(raw_row.get("action") or "").strip(),
                    authority_state=(raw_row.get("authority_state") or "").strip(),
                    replacement_path=(raw_row.get("replacement_path") or "").strip(),
                    archive_destination=(raw_row.get("archive_destination") or "").strip(),
                    git_tracked_expected=parse_manifest_bool(
                        raw_row.get("git_tracked_expected") or "",
                        column="git_tracked_expected",
                        row_path=path,
                    ),
                    reason=(raw_row.get("reason") or "").strip(),
                    notes=(raw_row.get("notes") or "").strip(),
                )
            )
    return tuple(rows)


def discover_scoped_directories(repo_root: Path) -> tuple[str, ...]:
    names = sorted(
        entry.name
        for entry in repo_root.iterdir()
        if entry.is_dir() and is_scoped_generated_artifact(entry.name)
    )
    return tuple(names)


def validate_archive_destination(row: CleanupManifestRow) -> None:
    if row.action == "archive":
        if not row.archive_destination:
            raise ManifestValidationError(
                f"{row.path}: archive rows must declare archive_destination"
            )
    elif row.archive_destination:
        raise ManifestValidationError(
            f"{row.path}: only archive rows may declare archive_destination"
        )
    if not row.archive_destination:
        return
    archive_path = Path(row.archive_destination)
    if archive_path.is_absolute():
        raise ManifestValidationError(
            f"{row.path}: archive_destination must be repo-relative"
        )
    archive_parts = archive_path.parts
    expected_parts = ARCHIVE_PREFIX.parts
    if archive_parts[: len(expected_parts)] != expected_parts:
        raise ManifestValidationError(
            f"{row.path}: archive_destination must stay under "
            f"{ARCHIVE_PREFIX.as_posix()}"
        )


def validate_manifest_row(repo_root: Path, row: CleanupManifestRow) -> None:
    path = Path(row.path)
    if path.is_absolute() or len(path.parts) != 1:
        raise ManifestValidationError(
            f"{row.path}: path must point to a single top-level directory"
        )
    if not is_scoped_generated_artifact(row.path):
        raise ManifestValidationError(
            f"{row.path}: path is outside the generated-artifact cleanup scope"
        )
    if row.action not in VALID_ACTIONS:
        raise ManifestValidationError(
            f"{row.path}: action must be one of {', '.join(VALID_ACTIONS)}"
        )
    if not row.artifact_family:
        raise ManifestValidationError(f"{row.path}: artifact_family is required")
    if not row.artifact_kind:
        raise ManifestValidationError(f"{row.path}: artifact_kind is required")
    if not row.authority_state:
        raise ManifestValidationError(f"{row.path}: authority_state is required")
    if not row.reason:
        raise ManifestValidationError(f"{row.path}: reason is required")
    validate_archive_destination(row)
    artifact_path = repo_root / row.path
    if not artifact_path.exists():
        raise ManifestValidationError(f"{row.path}: path does not exist")
    if not artifact_path.is_dir():
        raise ManifestValidationError(f"{row.path}: path is not a directory")
    if row.replacement_path:
        replacement_path = repo_root / row.replacement_path
        if not replacement_path.exists():
            raise ManifestValidationError(
                f"{row.path}: replacement_path does not exist: {row.replacement_path}"
            )


def get_git_tracked_status(repo_root: Path, relative_path: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--", relative_path],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    if completed.returncode != 0:
        return False
    return bool(completed.stdout.strip())


def get_directory_size_bytes(path: Path) -> int:
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def build_cleanup_dry_run_report(
    repo_root: Path,
    manifest_path: Path,
) -> CleanupDryRunReport:
    rows = load_cleanup_manifest(manifest_path)
    if not rows:
        raise ManifestValidationError("Manifest is empty")
    path_counts: dict[str, int] = {}
    for row in rows:
        path_counts[row.path] = path_counts.get(row.path, 0) + 1
    duplicates = sorted(path for path, count in path_counts.items() if count > 1)
    if duplicates:
        raise ManifestValidationError(
            "Manifest contains duplicate rows: " + ", ".join(duplicates)
        )
    for row in rows:
        validate_manifest_row(repo_root, row)
    scoped_dirs = set(discover_scoped_directories(repo_root))
    manifest_dirs = {row.path for row in rows}
    unclassified = sorted(scoped_dirs - manifest_dirs)
    if unclassified:
        raise ManifestValidationError(
            "Unclassified in-scope directories: " + ", ".join(unclassified)
        )
    extra_rows = sorted(manifest_dirs - scoped_dirs)
    if extra_rows:
        raise ManifestValidationError(
            "Manifest rows point at directories outside the current scope: "
            + ", ".join(extra_rows)
        )
    report_rows: list[CleanupReportRow] = []
    for row in sorted(rows, key=lambda item: item.path):
        tracked_actual = get_git_tracked_status(repo_root, row.path)
        report_rows.append(
            CleanupReportRow(
                path=row.path,
                artifact_family=row.artifact_family,
                artifact_kind=row.artifact_kind,
                action=row.action,
                authority_state=row.authority_state,
                replacement_path=row.replacement_path,
                archive_destination=row.archive_destination,
                git_tracked_expected=row.git_tracked_expected,
                git_tracked_actual=tracked_actual,
                git_tracked_mismatch=row.git_tracked_expected != tracked_actual,
                size_bytes=get_directory_size_bytes(repo_root / row.path),
                reason=row.reason,
                notes=row.notes,
            )
        )
    summaries: list[ActionSummary] = []
    for action in VALID_ACTIONS:
        action_rows = [row for row in report_rows if row.action == action]
        summaries.append(
            ActionSummary(
                action=action,
                directory_count=len(action_rows),
                total_size_bytes=sum(row.size_bytes for row in action_rows),
                tracked_count=sum(1 for row in action_rows if row.git_tracked_actual),
                untracked_count=sum(1 for row in action_rows if not row.git_tracked_actual),
                mismatch_count=sum(1 for row in action_rows if row.git_tracked_mismatch),
            )
        )
    return CleanupDryRunReport(
        repo_root=str(repo_root.resolve()),
        manifest_path=str(manifest_path.resolve()),
        summaries=tuple(summaries),
        rows=tuple(report_rows),
    )


def format_bytes(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f}"


def _filtered_rows(
    report: CleanupDryRunReport, show: str
) -> tuple[CleanupReportRow, ...]:
    if show == "all":
        return report.rows
    return tuple(row for row in report.rows if row.action == show)


def format_cleanup_report_table(report: CleanupDryRunReport, *, show: str) -> str:
    lines = [
        f"Cleanup dry-run repo_root={report.repo_root}",
        f"Manifest={report.manifest_path}",
        "",
        "Summary by action",
        "action    dirs  size_mb  tracked  untracked  mismatch",
    ]
    for summary in report.summaries:
        lines.append(
            f"{summary.action:<8}  "
            f"{summary.directory_count:>4}  "
            f"{format_bytes(summary.total_size_bytes):>7}  "
            f"{summary.tracked_count:>7}  "
            f"{summary.untracked_count:>9}  "
            f"{summary.mismatch_count:>8}"
        )
    rows = _filtered_rows(report, show)
    lines.extend(
        [
            "",
            f"Detailed rows show={show}",
            "path | action | tracked_expected | tracked_actual | mismatch | size_mb | replacement_path",
        ]
    )
    for row in rows:
        replacement = row.replacement_path or "-"
        lines.append(
            " | ".join(
                [
                    row.path,
                    row.action,
                    str(row.git_tracked_expected).lower(),
                    str(row.git_tracked_actual).lower(),
                    str(row.git_tracked_mismatch).lower(),
                    format_bytes(row.size_bytes),
                    replacement,
                ]
            )
        )
    return "\n".join(lines)


def format_cleanup_report_json(report: CleanupDryRunReport, *, show: str) -> str:
    payload = {
        "repo_root": report.repo_root,
        "manifest_path": report.manifest_path,
        "summaries": [asdict(summary) for summary in report.summaries],
        "rows": [asdict(row) for row in _filtered_rows(report, show)],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run validator for generated curtailment artifact cleanup."
    )
    parser.add_argument(
        "--manifest-path",
        default="cleanup/curtailment_generated_artifacts_manifest_v1.csv",
        help="Path to the cleanup manifest. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root to scan for scoped generated artifact directories.",
    )
    parser.add_argument(
        "--show",
        choices=VALID_SHOW,
        default="all",
        help="Filter the detailed row output by action.",
    )
    parser.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default="table",
        help="Output format for the dry-run report.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    try:
        report = build_cleanup_dry_run_report(repo_root, manifest_path)
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.format == "json":
        print(format_cleanup_report_json(report, show=args.show))
        return 0
    print(format_cleanup_report_table(report, show=args.show))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
