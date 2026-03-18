import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cleanup.dry_run_generated_artifact_cleanup import (
    ManifestValidationError,
    build_cleanup_dry_run_report,
    format_cleanup_report_json,
)


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


class GeneratedArtifactCleanupTests(unittest.TestCase):
    def test_build_cleanup_dry_run_report_accepts_custom_scope_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "bmu_truth_history_phase6_remit_fix"
            keep_dir.mkdir()
            (keep_dir / "fact.csv").write_text("x\n", encoding="utf-8")
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
                    )
                ],
            )
            with patch(
                "cleanup.dry_run_generated_artifact_cleanup.get_git_tracked_status",
                return_value=True,
            ):
                report = build_cleanup_dry_run_report(
                    root,
                    manifest_path,
                    scope_prefixes=("bmu_",),
                )
            self.assertEqual(len(report.rows), 1)
            self.assertEqual(report.rows[0].path, keep_dir.name)

    def test_build_cleanup_dry_run_report_with_mixed_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "model_readiness_dec_2024_event_lifecycle"
            archive_dir = root / "model_readiness_dec_2024_opening_guardrail"
            delete_dir = root / "benchmark_window_scouts"
            keep_dir.mkdir()
            archive_dir.mkdir()
            delete_dir.mkdir()
            (keep_dir / "fact.csv").write_text("x\n", encoding="utf-8")
            (archive_dir / "fact.csv").write_text("yy\n", encoding="utf-8")
            (delete_dir / "fact.csv").write_text("zzz\n", encoding="utf-8")
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        keep_dir.name,
                        action="keep",
                        authority_state="authoritative_readiness",
                    ),
                    _row(
                        archive_dir.name,
                        action="archive",
                        replacement_path=keep_dir.name,
                        archive_destination=(
                            "_local_archive/generated_outputs/"
                            "model_readiness_dec_2024_opening_guardrail"
                        ),
                        authority_state="superseded_readiness_step",
                    ),
                    _row(
                        delete_dir.name,
                        action="delete",
                        git_tracked_expected="false",
                        artifact_family="scout_output",
                        artifact_kind="scout_batch",
                        authority_state="disposable_local_screen",
                    ),
                ],
            )
            tracked_map = {
                keep_dir.name: True,
                archive_dir.name: True,
                delete_dir.name: False,
            }
            with patch(
                "cleanup.dry_run_generated_artifact_cleanup.get_git_tracked_status",
                side_effect=lambda _root, rel_path: tracked_map[rel_path],
            ):
                report = build_cleanup_dry_run_report(root, manifest_path)
            summaries = {summary.action: summary for summary in report.summaries}
            self.assertEqual(summaries["keep"].directory_count, 1)
            self.assertEqual(summaries["archive"].directory_count, 1)
            self.assertEqual(summaries["delete"].directory_count, 1)
            self.assertEqual(summaries["keep"].tracked_count, 1)
            self.assertEqual(summaries["delete"].untracked_count, 1)
            self.assertFalse(any(row.git_tracked_mismatch for row in report.rows))
            json_text = format_cleanup_report_json(report, show="all")
            self.assertIn('"action": "archive"', json_text)

    def test_build_cleanup_dry_run_report_fails_on_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        "model_readiness_dec_2024_event_lifecycle",
                        action="keep",
                    )
                ],
            )
            with self.assertRaisesRegex(
                ManifestValidationError,
                "path does not exist",
            ):
                build_cleanup_dry_run_report(root, manifest_path)

    def test_build_cleanup_dry_run_report_fails_on_unclassified_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "model_readiness_dec_2024_event_lifecycle"
            extra_dir = root / "model_readiness_dec_2024_opening_guardrail"
            keep_dir.mkdir()
            extra_dir.mkdir()
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        keep_dir.name,
                        action="keep",
                    )
                ],
            )
            with patch(
                "cleanup.dry_run_generated_artifact_cleanup.get_git_tracked_status",
                return_value=True,
            ):
                with self.assertRaisesRegex(
                    ManifestValidationError,
                    "Unclassified in-scope directories",
                ):
                    build_cleanup_dry_run_report(root, manifest_path)

    def test_build_cleanup_dry_run_report_fails_on_bad_archive_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_dir = root / "model_readiness_dec_2024_opening_guardrail"
            replacement_dir = root / "model_readiness_dec_2024_event_lifecycle"
            archive_dir.mkdir()
            replacement_dir.mkdir()
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        archive_dir.name,
                        action="archive",
                        replacement_path=replacement_dir.name,
                        archive_destination="archive/model_readiness_dec_2024_opening_guardrail",
                    ),
                    _row(
                        replacement_dir.name,
                        action="keep",
                    ),
                ],
            )
            with self.assertRaisesRegex(
                ManifestValidationError,
                "archive_destination must stay under",
            ):
                build_cleanup_dry_run_report(root, manifest_path)

    def test_build_cleanup_dry_run_report_fails_on_missing_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_dir = root / "model_readiness_dec_2024_opening_guardrail"
            archive_dir.mkdir()
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        archive_dir.name,
                        action="archive",
                        replacement_path="model_readiness_dec_2024_event_lifecycle",
                        archive_destination=(
                            "_local_archive/generated_outputs/"
                            "model_readiness_dec_2024_opening_guardrail"
                        ),
                    )
                ],
            )
            with self.assertRaisesRegex(
                ManifestValidationError,
                "replacement_path does not exist",
            ):
                build_cleanup_dry_run_report(root, manifest_path)

    def test_build_cleanup_dry_run_report_reports_git_tracking_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keep_dir = root / "model_readiness_dec_2024_event_lifecycle"
            keep_dir.mkdir()
            manifest_path = root / "manifest.csv"
            _write_manifest(
                manifest_path,
                [
                    _row(
                        keep_dir.name,
                        action="keep",
                        git_tracked_expected="false",
                    )
                ],
            )
            with patch(
                "cleanup.dry_run_generated_artifact_cleanup.get_git_tracked_status",
                return_value=True,
            ):
                report = build_cleanup_dry_run_report(root, manifest_path)
            self.assertTrue(report.rows[0].git_tracked_mismatch)
            summaries = {summary.action: summary for summary in report.summaries}
            self.assertEqual(summaries["keep"].mismatch_count, 1)
