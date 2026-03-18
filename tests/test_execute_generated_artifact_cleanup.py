import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cleanup.execute_generated_artifact_cleanup import (
    FINAL_REVIEW_TOKEN,
    build_execution_plan,
    execute_cleanup_plan,
    main,
)
from cleanup.dry_run_generated_artifact_cleanup import ManifestValidationError


MANIFEST_COLUMNS = (
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


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    path: str,
    *,
    action: str,
    replacement_path: str = "",
    archive_destination: str = "",
    git_tracked_expected: str = "true",
    artifact_family: str = "readiness_run",
    artifact_kind: str = "snapshot",
    authority_state: str = "test_state",
    reason: str = "test row",
    notes: str = "",
) -> dict[str, str]:
    return {
        "path": path,
        "artifact_family": artifact_family,
        "artifact_kind": artifact_kind,
        "action": action,
        "authority_state": authority_state,
        "replacement_path": replacement_path,
        "archive_destination": archive_destination,
        "git_tracked_expected": git_tracked_expected,
        "reason": reason,
        "notes": notes,
    }


class ExecuteGeneratedArtifactCleanupTests(unittest.TestCase):
    def test_build_execution_plan_accepts_custom_scope_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "bmu_truth_history_phase6_remit_fix"
            archive_dir = root / "bmu_truth_history_phase5_targeted_rerun"
            keep_dir.mkdir()
            archive_dir.mkdir()
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        keep_dir.name,
                        action="keep",
                        artifact_family="bmu_truth",
                        artifact_kind="truth_phase_authoritative",
                        authority_state="authoritative_truth_snapshot",
                    ),
                    _row(
                        archive_dir.name,
                        action="archive",
                        artifact_family="bmu_truth",
                        artifact_kind="truth_phase_snapshot",
                        authority_state="superseded_truth_phase",
                        replacement_path=keep_dir.name,
                        archive_destination=(
                            "_local_archive/generated_outputs/"
                            "bmu_truth_history_phase5_targeted_rerun"
                        ),
                    ),
                ],
            )
            with patch(
                "cleanup.execute_generated_artifact_cleanup.get_git_tracked_status",
                return_value=True,
            ):
                result = build_execution_plan(
                    root,
                    manifest_path,
                    action="archive",
                    scope_prefixes=("bmu_",),
                )
            self.assertEqual(len(result.rows), 1)
            self.assertEqual(result.rows[0].path, archive_dir.name)

    def test_build_execution_plan_filters_archive_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "model_readiness_dec_2024_event_lifecycle"
            archive_dir = root / "model_readiness_dec_2024_opening_guardrail"
            keep_dir.mkdir()
            archive_dir.mkdir()
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(keep_dir.name, action="keep", authority_state="authoritative"),
                    _row(
                        archive_dir.name,
                        action="archive",
                        replacement_path=keep_dir.name,
                        archive_destination=(
                            "_local_archive/generated_outputs/"
                            "model_readiness_dec_2024_opening_guardrail"
                        ),
                        authority_state="superseded",
                    ),
                ],
            )
            with patch(
                "cleanup.execute_generated_artifact_cleanup.get_git_tracked_status",
                return_value=True,
            ):
                result = build_execution_plan(root, manifest_path, action="archive")
            self.assertEqual(result.selected_action, "archive")
            self.assertEqual(len(result.rows), 1)
            self.assertEqual(result.rows[0].path, archive_dir.name)
            self.assertTrue(
                result.rows[0].destination_path.endswith(
                    "_local_archive\\generated_outputs\\model_readiness_dec_2024_opening_guardrail"
                )
            )
            self.assertFalse(result.executed)

    def test_execute_cleanup_plan_moves_archive_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "model_readiness_dec_2024_event_lifecycle"
            archive_dir = root / "model_readiness_dec_2024_opening_guardrail"
            keep_dir.mkdir()
            archive_dir.mkdir()
            (archive_dir / "fact.csv").write_text("x\n", encoding="utf-8")
            manifest_path = root / "manifest.csv"
            archive_destination = (
                "_local_archive/generated_outputs/model_readiness_dec_2024_opening_guardrail"
            )
            _write_manifest(
                manifest_path,
                [
                    _row(keep_dir.name, action="keep", authority_state="authoritative"),
                    _row(
                        archive_dir.name,
                        action="archive",
                        replacement_path=keep_dir.name,
                        archive_destination=archive_destination,
                        authority_state="superseded",
                    ),
                ],
            )
            with patch(
                "cleanup.execute_generated_artifact_cleanup.get_git_tracked_status",
                return_value=True,
            ):
                result = build_execution_plan(root, manifest_path, action="archive")
            executed = execute_cleanup_plan(result)
            self.assertTrue(executed.executed)
            self.assertFalse(archive_dir.exists())
            self.assertTrue((root / archive_destination / "fact.csv").exists())

    def test_execute_cleanup_plan_removes_delete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            delete_dir = root / "benchmark_window_scouts"
            keep_dir = root / "model_readiness_dec_2024_event_lifecycle"
            delete_dir.mkdir()
            keep_dir.mkdir()
            (delete_dir / "fact.csv").write_text("x\n", encoding="utf-8")
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        delete_dir.name,
                        action="delete",
                        artifact_family="scout_output",
                        artifact_kind="scout_batch",
                        authority_state="disposable",
                    ),
                    _row(keep_dir.name, action="keep", authority_state="authoritative"),
                ],
            )
            with patch(
                "cleanup.execute_generated_artifact_cleanup.get_git_tracked_status",
                return_value=False,
            ):
                result = build_execution_plan(root, manifest_path, action="delete")
            self.assertEqual(result.rows[0].destination_path, "")
            executed = execute_cleanup_plan(result)
            self.assertTrue(executed.executed)
            self.assertFalse(delete_dir.exists())
            self.assertTrue(keep_dir.exists())

    def test_execute_cleanup_plan_fails_if_archive_destination_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "model_readiness_dec_2024_event_lifecycle"
            archive_dir = root / "model_readiness_dec_2024_opening_guardrail"
            keep_dir.mkdir()
            archive_dir.mkdir()
            archive_destination = root / (
                "_local_archive/generated_outputs/model_readiness_dec_2024_opening_guardrail"
            )
            archive_destination.mkdir(parents=True)
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(keep_dir.name, action="keep", authority_state="authoritative"),
                    _row(
                        archive_dir.name,
                        action="archive",
                        replacement_path=keep_dir.name,
                        archive_destination=str(archive_destination.relative_to(root)),
                        authority_state="superseded",
                    ),
                ],
            )
            with patch(
                "cleanup.execute_generated_artifact_cleanup.get_git_tracked_status",
                return_value=True,
            ):
                result = build_execution_plan(root, manifest_path, action="archive")
            with self.assertRaisesRegex(
                ManifestValidationError,
                "archive destination already exists",
            ):
                execute_cleanup_plan(result)

    def test_main_refuses_execute_without_final_review_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            delete_dir = root / "benchmark_window_scouts"
            delete_dir.mkdir()
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        delete_dir.name,
                        action="delete",
                        artifact_family="scout_output",
                        artifact_kind="scout_batch",
                        authority_state="disposable",
                    )
                ],
            )
            with patch(
                "cleanup.execute_generated_artifact_cleanup.get_git_tracked_status",
                return_value=False,
            ):
                exit_code = main(
                    [
                        "--repo-root",
                        str(root),
                        "--manifest-path",
                        str(manifest_path),
                        "--action",
                        "delete",
                        "--execute",
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertTrue(delete_dir.exists())

    def test_main_executes_after_final_review_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            delete_dir = root / "benchmark_window_scouts"
            delete_dir.mkdir()
            (delete_dir / "fact.csv").write_text("x\n", encoding="utf-8")
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        delete_dir.name,
                        action="delete",
                        artifact_family="scout_output",
                        artifact_kind="scout_batch",
                        authority_state="disposable",
                    )
                ],
            )
            with patch(
                "cleanup.execute_generated_artifact_cleanup.get_git_tracked_status",
                return_value=False,
            ):
                exit_code = main(
                    [
                        "--repo-root",
                        str(root),
                        "--manifest-path",
                        str(manifest_path),
                        "--action",
                        "delete",
                        "--execute",
                        "--confirm-final-review",
                        FINAL_REVIEW_TOKEN,
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertFalse(delete_dir.exists())
