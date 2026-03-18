from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from cleanup.dry_run_generated_artifact_cleanup import (
        CleanupManifestRow,
        ManifestValidationError,
        get_directory_size_bytes,
        get_git_tracked_status,
        load_cleanup_manifest,
        normalize_scope_prefixes,
        resolve_repo_path,
        validate_manifest_row,
    )
except ModuleNotFoundError:
    from dry_run_generated_artifact_cleanup import (  # type: ignore
        CleanupManifestRow,
        ManifestValidationError,
        get_directory_size_bytes,
        get_git_tracked_status,
        load_cleanup_manifest,
        normalize_scope_prefixes,
        resolve_repo_path,
        validate_manifest_row,
    )

FINAL_REVIEW_TOKEN = "reviewed-dry-run"
VALID_ACTIONS = ("archive", "delete")
VALID_FORMATS = ("table", "json")


@dataclass(frozen=True)
class CleanupExecutionRow:
    path: str
    action: str
    destination_path: str
    size_bytes: int
    git_tracked_expected: bool
    git_tracked_actual: bool
    git_tracked_mismatch: bool


@dataclass(frozen=True)
class CleanupExecutionSummary:
    action: str
    directory_count: int
    total_size_bytes: int
    tracked_count: int
    untracked_count: int
    mismatch_count: int


@dataclass(frozen=True)
class CleanupExecutionResult:
    repo_root: str
    manifest_path: str
    summary: CleanupExecutionSummary
    executed: bool
    selected_action: str
    rows: tuple[CleanupExecutionRow, ...]


def _destination_for_row(repo_root: Path, row: CleanupManifestRow) -> Path | None:
    if row.action == "archive":
        if not row.archive_destination:
            raise ManifestValidationError(
                f"{row.path}: archive row is missing archive_destination"
            )
        return repo_root / row.archive_destination
    return None


def build_execution_plan(
    repo_root: Path,
    manifest_path: Path,
    *,
    action: str,
    scope_prefixes: tuple[str, ...] | list[str] | None = None,
) -> CleanupExecutionResult:
    if action not in VALID_ACTIONS:
        raise ManifestValidationError(
            f"action must be one of {', '.join(VALID_ACTIONS)}"
        )
    normalized_scope_prefixes = normalize_scope_prefixes(scope_prefixes)
    manifest_rows = load_cleanup_manifest(manifest_path)
    path_counts: dict[str, int] = {}
    for row in manifest_rows:
        path_counts[row.path] = path_counts.get(row.path, 0) + 1
    duplicates = sorted(path for path, count in path_counts.items() if count > 1)
    if duplicates:
        raise ManifestValidationError(
            "Manifest contains duplicate rows: " + ", ".join(duplicates)
        )
    selected_rows = tuple(row for row in manifest_rows if row.action == action)
    rows: list[CleanupExecutionRow] = []
    for row in selected_rows:
        validate_manifest_row(
            repo_root,
            row,
            scope_prefixes=normalized_scope_prefixes,
        )
        tracked_actual = get_git_tracked_status(repo_root, row.path)
        destination = _destination_for_row(repo_root, row)
        rows.append(
            CleanupExecutionRow(
                path=row.path,
                action=row.action,
                destination_path=str(destination) if destination is not None else "",
                size_bytes=get_directory_size_bytes(repo_root / row.path),
                git_tracked_expected=row.git_tracked_expected,
                git_tracked_actual=tracked_actual,
                git_tracked_mismatch=row.git_tracked_expected != tracked_actual,
            )
        )
    summary = CleanupExecutionSummary(
        action=action,
        directory_count=len(rows),
        total_size_bytes=sum(row.size_bytes for row in rows),
        tracked_count=sum(1 for row in rows if row.git_tracked_actual),
        untracked_count=sum(1 for row in rows if not row.git_tracked_actual),
        mismatch_count=sum(1 for row in rows if row.git_tracked_mismatch),
    )
    return CleanupExecutionResult(
        repo_root=str(repo_root.resolve()),
        manifest_path=str(manifest_path.resolve()),
        summary=summary,
        executed=False,
        selected_action=action,
        rows=tuple(rows),
    )


def format_execution_result_table(result: CleanupExecutionResult) -> str:
    lines = [
        f"Execution plan repo_root={result.repo_root}",
        f"Manifest={result.manifest_path}",
        "",
        "Selected action summary",
        "action | dirs | size_mb | tracked | untracked | mismatch | executed",
        " | ".join(
            [
                result.summary.action,
                str(result.summary.directory_count),
                f"{result.summary.total_size_bytes / (1024 * 1024):.2f}",
                str(result.summary.tracked_count),
                str(result.summary.untracked_count),
                str(result.summary.mismatch_count),
                str(result.executed).lower(),
            ]
        ),
        "",
        f"Execution plan action={result.selected_action} executed={str(result.executed).lower()}",
        "path | action | destination_path | size_mb | tracked_expected | tracked_actual | mismatch",
    ]
    for row in result.rows:
        destination = row.destination_path or "-"
        lines.append(
            " | ".join(
                [
                    row.path,
                    row.action,
                    destination,
                    f"{row.size_bytes / (1024 * 1024):.2f}",
                    str(row.git_tracked_expected).lower(),
                    str(row.git_tracked_actual).lower(),
                    str(row.git_tracked_mismatch).lower(),
                ]
            )
        )
    return "\n".join(lines)


def format_execution_result_json(result: CleanupExecutionResult) -> str:
    payload = {
        "repo_root": result.repo_root,
        "manifest_path": result.manifest_path,
        "selected_action": result.selected_action,
        "executed": result.executed,
        "summary": asdict(result.summary),
        "rows": [asdict(row) for row in result.rows],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _execute_archive_row(repo_root: Path, row: CleanupExecutionRow) -> None:
    source = repo_root / row.path
    destination = Path(row.destination_path)
    if destination.exists():
        raise ManifestValidationError(
            f"{row.path}: archive destination already exists: {destination}"
        )
    _ensure_parent(destination)
    shutil.move(str(source), str(destination))


def _execute_delete_row(repo_root: Path, row: CleanupExecutionRow) -> None:
    source = repo_root / row.path
    if source.exists():
        shutil.rmtree(source)


def execute_cleanup_plan(result: CleanupExecutionResult) -> CleanupExecutionResult:
    repo_root = Path(result.repo_root)
    for row in result.rows:
        if row.action == "archive":
            _execute_archive_row(repo_root, row)
        elif row.action == "delete":
            _execute_delete_row(repo_root, row)
        else:
            raise ManifestValidationError(f"{row.path}: unsupported action {row.action}")
    return CleanupExecutionResult(
        repo_root=result.repo_root,
        manifest_path=result.manifest_path,
        summary=result.summary,
        executed=True,
        selected_action=result.selected_action,
        rows=result.rows,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute generated curtailment artifact cleanup after a validated final dry-run review."
        )
    )
    parser.add_argument(
        "--manifest-path",
        default="cleanup/curtailment_generated_artifacts_manifest_v1.csv",
        help="Path to the cleanup manifest. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root to scan and mutate.",
    )
    parser.add_argument(
        "--scope-prefix",
        action="append",
        default=[],
        help=(
            "Optional top-level directory prefix to define cleanup scope. "
            "Repeat for multiple families. Defaults to the curtailment scope."
        ),
    )
    parser.add_argument(
        "--action",
        choices=VALID_ACTIONS,
        required=True,
        help="Which manifest action to execute.",
    )
    parser.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform archive or delete after validation.",
    )
    parser.add_argument(
        "--confirm-final-review",
        default="",
        help=(
            "Required with --execute. Must equal "
            f"{FINAL_REVIEW_TOKEN!r} to confirm the final dry-run was reviewed."
        ),
    )
    return parser.parse_args(argv)


def _render_result(result: CleanupExecutionResult, *, output_format: str) -> str:
    if output_format == "json":
        return format_execution_result_json(result)
    return format_execution_result_table(result)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    manifest_path = resolve_repo_path(repo_root, args.manifest_path)
    scope_prefixes = tuple(args.scope_prefix) if args.scope_prefix else None
    try:
        result = build_execution_plan(
            repo_root,
            manifest_path,
            action=args.action,
            scope_prefixes=scope_prefixes,
        )
        if args.execute:
            if args.confirm_final_review != FINAL_REVIEW_TOKEN:
                raise ManifestValidationError(
                    "Refusing to execute without --confirm-final-review "
                    f"{FINAL_REVIEW_TOKEN!r}"
                )
            result = execute_cleanup_plan(result)
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(_render_result(result, output_format=args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
