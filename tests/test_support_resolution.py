import tempfile
import unittest
from pathlib import Path

import pandas as pd

from history_store import upsert_frame_to_sqlite
from support_resolution import (
    SUPPORT_CASE_RESOLUTION_TABLE,
    SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE,
    SUPPORT_RERUN_CANDIDATE_DAILY_TABLE,
    SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE,
    SUPPORT_RERUN_GATE_BATCH_TABLE,
    SUPPORT_RERUN_GATE_DAILY_TABLE,
    SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE,
    SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE,
    SUPPORT_RESOLUTION_BATCH_TABLE,
    SUPPORT_RESOLUTION_DAILY_TABLE,
    annotate_support_resolution_pattern,
    annotate_support_case_resolution,
    materialize_truth_store_support_resolution,
    read_support_case_resolution,
    read_support_resolution_pattern_review,
    read_support_rerun_candidate_review,
    read_support_rerun_gate_review,
    read_support_resolution_review,
)


class SupportResolutionTests(unittest.TestCase):
    def _support_case_family_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "support_batch_id": "support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                    "support_case_family_key": "support_fail_warn_days7_families5_2024-10-02_2024-10-06:2024-10-02:HOWBO",
                    "support_generated_at_utc": "2026-03-11T12:00:00Z",
                    "support_status_mode": "fail_warn",
                    "support_top_days": 7,
                    "support_top_families_per_day": 5,
                    "support_case_day_rank": 1,
                    "support_case_family_rank": 1,
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "publication_anomaly_next_action": "support_query_bod_sentinel_and_boalf_publication",
                    "bmu_family_key": "HOWBO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "publication_anomaly_candidate_mwh_lower_bound": 50.0,
                    "publication_anomaly_share_of_day_total": 0.6,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.5,
                    "publication_anomaly_dominant_state": "sentinel_bod_present",
                    "day_family_rank_by_publication_anomaly": 1,
                    "support_question_code": "query_bod_sentinel_and_missing_boalf",
                    "support_recommended_action": "ask_elexon_about_suspect_bod_sentinel_and_missing_published_boalf",
                    "distinct_bmu_count": 1,
                    "half_hour_count": 48,
                    "published_boalf_absent_half_hour_count": 48,
                    "physical_without_boalf_half_hour_count": 48,
                    "physical_without_boalf_negative_bid_half_hour_count": 0,
                    "physical_without_boalf_sentinel_half_hour_count": 48,
                    "availability_like_dynamic_limit_half_hour_count": 0,
                    "most_negative_bid_gbp_per_mwh": -9999.0,
                    "publication_audit_priority_rank": 1,
                    "publication_audit_state": "physical_without_boalf",
                },
                {
                    "support_batch_id": "support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                    "support_case_family_key": "support_fail_warn_days7_families5_2024-10-02_2024-10-06:2024-10-06:HOWAO",
                    "support_generated_at_utc": "2026-03-11T12:00:00Z",
                    "support_status_mode": "fail_warn",
                    "support_top_days": 7,
                    "support_top_families_per_day": 5,
                    "support_case_day_rank": 2,
                    "support_case_family_rank": 1,
                    "settlement_date": "2024-10-06",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "publication_anomaly_next_action": "support_query_missing_published_boalf",
                    "bmu_family_key": "HOWAO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "publication_anomaly_candidate_mwh_lower_bound": 40.0,
                    "publication_anomaly_share_of_day_total": 0.5,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.4,
                    "publication_anomaly_dominant_state": "negative_bid_without_boalf",
                    "day_family_rank_by_publication_anomaly": 1,
                    "support_question_code": "query_missing_boalf_with_negative_bid_and_physical_gap",
                    "support_recommended_action": "ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf",
                    "distinct_bmu_count": 1,
                    "half_hour_count": 48,
                    "published_boalf_absent_half_hour_count": 48,
                    "physical_without_boalf_half_hour_count": 48,
                    "physical_without_boalf_negative_bid_half_hour_count": 48,
                    "physical_without_boalf_sentinel_half_hour_count": 0,
                    "availability_like_dynamic_limit_half_hour_count": 0,
                    "most_negative_bid_gbp_per_mwh": -75.0,
                    "publication_audit_priority_rank": 1,
                    "publication_audit_state": "physical_without_boalf_negative_bid",
                },
            ]
        )

    def _support_case_pattern_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "support_batch_id": "support_fail_warn_days7_families5_2024-10-01_2024-10-07",
                    "support_case_family_key": "support_fail_warn_days7_families5_2024-10-01_2024-10-07:2024-10-01:HOWAO",
                    "support_generated_at_utc": "2026-03-11T12:00:00Z",
                    "support_status_mode": "fail_warn",
                    "support_top_days": 7,
                    "support_top_families_per_day": 5,
                    "support_case_day_rank": 1,
                    "support_case_family_rank": 1,
                    "settlement_date": "2024-10-01",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "publication_anomaly_next_action": "support_query_missing_published_boalf",
                    "bmu_family_key": "HOWAO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "publication_anomaly_candidate_mwh_lower_bound": 20.0,
                    "publication_anomaly_share_of_day_total": 0.5,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.5,
                    "publication_anomaly_dominant_state": "negative_bid_without_boalf",
                    "day_family_rank_by_publication_anomaly": 1,
                    "support_question_code": "query_missing_boalf_with_negative_bid_and_physical_gap",
                    "support_recommended_action": "ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf",
                },
                {
                    "support_batch_id": "support_fail_warn_days7_families5_2024-10-01_2024-10-07",
                    "support_case_family_key": "support_fail_warn_days7_families5_2024-10-01_2024-10-07:2024-10-03:HOWAO",
                    "support_generated_at_utc": "2026-03-11T12:00:00Z",
                    "support_status_mode": "fail_warn",
                    "support_top_days": 7,
                    "support_top_families_per_day": 5,
                    "support_case_day_rank": 2,
                    "support_case_family_rank": 1,
                    "settlement_date": "2024-10-03",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "publication_anomaly_next_action": "support_query_missing_published_boalf",
                    "bmu_family_key": "HOWAO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "publication_anomaly_candidate_mwh_lower_bound": 18.0,
                    "publication_anomaly_share_of_day_total": 0.45,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.4,
                    "publication_anomaly_dominant_state": "negative_bid_without_boalf",
                    "day_family_rank_by_publication_anomaly": 1,
                    "support_question_code": "query_missing_boalf_with_negative_bid_and_physical_gap",
                    "support_recommended_action": "ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf",
                },
                {
                    "support_batch_id": "support_fail_warn_days7_families5_2024-10-01_2024-10-07",
                    "support_case_family_key": "support_fail_warn_days7_families5_2024-10-01_2024-10-07:2024-10-03:MOWEO",
                    "support_generated_at_utc": "2026-03-11T12:00:00Z",
                    "support_status_mode": "fail_warn",
                    "support_top_days": 7,
                    "support_top_families_per_day": 5,
                    "support_case_day_rank": 2,
                    "support_case_family_rank": 2,
                    "settlement_date": "2024-10-03",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "publication_anomaly_next_action": "support_query_missing_published_boalf",
                    "bmu_family_key": "MOWEO",
                    "bmu_family_label": "Moray East",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                    "mapping_status": "mapped",
                    "publication_anomaly_candidate_mwh_lower_bound": 12.0,
                    "publication_anomaly_share_of_day_total": 0.3,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.2,
                    "publication_anomaly_dominant_state": "negative_bid_without_boalf",
                    "day_family_rank_by_publication_anomaly": 2,
                    "support_question_code": "query_missing_boalf_with_negative_bid_and_physical_gap",
                    "support_recommended_action": "ask_elexon_why_physical_gap_and_negative_bid_exist_without_published_boalf",
                },
            ]
        )

    def test_materialize_support_resolution_defaults_to_open(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            support_case_family = self._support_case_family_frame()
            upsert_frame_to_sqlite(
                db_path,
                "fact_support_case_family_daily",
                support_case_family,
                ["support_batch_id", "settlement_date", "bmu_family_key"],
            )

            materialized = materialize_truth_store_support_resolution(
                db_path=db_path,
                generated_at_utc="2026-03-11T12:00:00Z",
            )
            resolution = materialized[SUPPORT_CASE_RESOLUTION_TABLE]

            self.assertEqual(len(resolution), 2)
            self.assertTrue((resolution["resolution_state"] == "open").all())
            self.assertTrue((resolution["truth_policy_action"] == "keep_out_of_precision").all())
            self.assertEqual(len(materialized[SUPPORT_RESOLUTION_DAILY_TABLE]), 2)
            self.assertEqual(materialized[SUPPORT_RESOLUTION_BATCH_TABLE].iloc[0]["support_resolution_state"], "blocked_by_open_cases")
            self.assertEqual(
                materialized[SUPPORT_RERUN_GATE_BATCH_TABLE].iloc[0]["support_rerun_gate_state"],
                "blocked_by_open_cases",
            )
            priority = materialized[SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE]
            self.assertEqual(priority["bmu_family_key"].tolist(), ["HOWBO", "HOWAO"])
            self.assertEqual(priority["open_case_priority_rank"].tolist(), [1, 2])

    def test_annotate_support_resolution_persists_across_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            support_case_family = self._support_case_family_frame()
            upsert_frame_to_sqlite(
                db_path,
                "fact_support_case_family_daily",
                support_case_family,
                ["support_batch_id", "settlement_date", "bmu_family_key"],
            )
            materialize_truth_store_support_resolution(
                db_path=db_path,
                generated_at_utc="2026-03-11T12:00:00Z",
            )

            annotated = annotate_support_case_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                settlement_date="2024-10-02",
                bmu_family_key="HOWBO",
                resolution_state="confirmed_publication_gap",
                truth_policy_action="fix_source_and_rerun",
                resolution_note="Elexon confirmed a publication gap on this family-day.",
                source_reference="ticket-123",
                generated_at_utc="2026-03-11T13:00:00Z",
            )
            self.assertEqual(annotated.iloc[0]["resolution_state"], "confirmed_publication_gap")
            self.assertEqual(annotated.iloc[0]["truth_policy_action"], "fix_source_and_rerun")

            refreshed = materialize_truth_store_support_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                generated_at_utc="2026-03-11T14:00:00Z",
            )[SUPPORT_CASE_RESOLUTION_TABLE]
            howbo = refreshed[refreshed["bmu_family_key"] == "HOWBO"].iloc[0]
            howao = refreshed[refreshed["bmu_family_key"] == "HOWAO"].iloc[0]
            self.assertEqual(howbo["resolution_state"], "confirmed_publication_gap")
            self.assertEqual(howbo["truth_policy_action"], "fix_source_and_rerun")
            self.assertEqual(howbo["source_reference"], "ticket-123")
            self.assertEqual(howao["resolution_state"], "open")
            review = read_support_resolution_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
            )
            batch = review[SUPPORT_RESOLUTION_BATCH_TABLE].iloc[0]
            self.assertEqual(batch["support_resolution_state"], "blocked_by_open_cases")
            self.assertEqual(batch["support_resolution_next_action"], "await_support_resolution")
            gate = read_support_rerun_gate_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
            )
            daily_gate = gate[SUPPORT_RERUN_GATE_DAILY_TABLE]
            self.assertEqual(
                daily_gate.loc[daily_gate["settlement_date"] == "2024-10-02", "support_rerun_gate_state"].iloc[0],
                "candidate_targeted_rerun",
            )
            self.assertEqual(
                gate[SUPPORT_RERUN_GATE_BATCH_TABLE].iloc[0]["support_rerun_gate_state"],
                "blocked_by_open_cases",
            )
            candidates = read_support_rerun_candidate_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
            )
            candidate_daily = candidates[SUPPORT_RERUN_CANDIDATE_DAILY_TABLE]
            candidate_family = candidates[SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE]
            self.assertEqual(candidate_daily["settlement_date"].tolist(), ["2024-10-02"])
            self.assertEqual(candidate_family["bmu_family_key"].tolist(), ["HOWBO"])
            self.assertEqual(
                candidate_family.iloc[0]["rerun_candidate_action"],
                "rerun_after_source_fix",
            )

    def test_read_support_resolution_filters_open_and_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            support_case_family = self._support_case_family_frame()
            upsert_frame_to_sqlite(
                db_path,
                "fact_support_case_family_daily",
                support_case_family,
                ["support_batch_id", "settlement_date", "bmu_family_key"],
            )
            materialize_truth_store_support_resolution(
                db_path=db_path,
                generated_at_utc="2026-03-11T12:00:00Z",
            )
            annotate_support_case_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                settlement_date="2024-10-02",
                bmu_family_key="HOWBO",
                resolution_state="confirmed_source_artifact",
                truth_policy_action="close_no_change",
                generated_at_utc="2026-03-11T13:00:00Z",
            )

            open_rows = read_support_case_resolution(db_path=db_path, resolution_filter="open")
            resolved_rows = read_support_case_resolution(db_path=db_path, resolution_filter="resolved")

            self.assertEqual(open_rows["bmu_family_key"].tolist(), ["HOWAO"])
            self.assertEqual(resolved_rows["bmu_family_key"].tolist(), ["HOWBO"])
            self.assertEqual(resolved_rows.iloc[0]["resolution_state"], "confirmed_source_artifact")

    def test_rerun_gate_ready_for_rerun_after_all_cases_resolved_with_fix_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            support_case_family = self._support_case_family_frame()
            upsert_frame_to_sqlite(
                db_path,
                "fact_support_case_family_daily",
                support_case_family,
                ["support_batch_id", "settlement_date", "bmu_family_key"],
            )
            materialize_truth_store_support_resolution(
                db_path=db_path,
                generated_at_utc="2026-03-11T12:00:00Z",
            )
            annotate_support_case_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                settlement_date="2024-10-02",
                bmu_family_key="HOWBO",
                resolution_state="confirmed_publication_gap",
                truth_policy_action="fix_source_and_rerun",
                generated_at_utc="2026-03-11T13:00:00Z",
            )
            annotate_support_case_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                settlement_date="2024-10-06",
                bmu_family_key="HOWAO",
                resolution_state="confirmed_non_boalf_pattern",
                truth_policy_action="eligible_for_new_evidence_tier",
                generated_at_utc="2026-03-11T14:00:00Z",
            )

            gate = read_support_rerun_gate_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                gate_filter="ready_for_rerun",
            )
            batch = gate[SUPPORT_RERUN_GATE_BATCH_TABLE]
            daily = gate[SUPPORT_RERUN_GATE_DAILY_TABLE]
            priority = gate[SUPPORT_OPEN_CASE_PRIORITY_FAMILY_TABLE]

            self.assertEqual(batch.iloc[0]["support_rerun_gate_state"], "ready_for_targeted_rerun")
            self.assertEqual(int(batch.iloc[0]["candidate_rerun_day_count"]), 2)
            self.assertEqual(daily["support_rerun_gate_state"].tolist(), ["candidate_targeted_rerun", "candidate_targeted_rerun"])
            self.assertTrue(priority.empty)

    def test_rerun_gate_no_rerun_required_when_only_policy_lock_actions_remain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            support_case_family = self._support_case_family_frame()
            upsert_frame_to_sqlite(
                db_path,
                "fact_support_case_family_daily",
                support_case_family,
                ["support_batch_id", "settlement_date", "bmu_family_key"],
            )
            materialize_truth_store_support_resolution(
                db_path=db_path,
                generated_at_utc="2026-03-11T12:00:00Z",
            )
            annotate_support_case_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                settlement_date="2024-10-02",
                bmu_family_key="HOWBO",
                resolution_state="confirmed_non_boalf_pattern",
                truth_policy_action="keep_out_of_precision",
                generated_at_utc="2026-03-11T13:00:00Z",
            )
            annotate_support_case_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                settlement_date="2024-10-06",
                bmu_family_key="HOWAO",
                resolution_state="confirmed_source_artifact",
                truth_policy_action="close_no_change",
                generated_at_utc="2026-03-11T14:00:00Z",
            )

            gate = read_support_rerun_gate_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
                gate_filter="no_rerun_required",
            )
            batch = gate[SUPPORT_RERUN_GATE_BATCH_TABLE]
            daily = gate[SUPPORT_RERUN_GATE_DAILY_TABLE]
            self.assertEqual(batch.iloc[0]["support_rerun_gate_state"], "no_rerun_required")
            self.assertEqual(batch.iloc[0]["support_rerun_next_action"], "lock_truth_policy_no_rerun")
            self.assertEqual(daily["support_rerun_gate_state"].tolist(), ["candidate_policy_lock", "candidate_policy_lock"])
            candidates = read_support_rerun_candidate_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-06",
            )
            self.assertTrue(candidates[SUPPORT_RERUN_CANDIDATE_DAILY_TABLE].empty)
            self.assertTrue(candidates[SUPPORT_RERUN_CANDIDATE_FAMILY_TABLE].empty)

    def test_support_resolution_pattern_summary_groups_repeated_open_family_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            support_case_family = self._support_case_pattern_frame()
            upsert_frame_to_sqlite(
                db_path,
                "fact_support_case_family_daily",
                support_case_family,
                ["support_batch_id", "settlement_date", "bmu_family_key"],
            )
            materialize_truth_store_support_resolution(
                db_path=db_path,
                generated_at_utc="2026-03-11T12:00:00Z",
            )

            pattern_frames = read_support_resolution_pattern_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-01_2024-10-07",
                pattern_filter="multi_day",
            )
            summary = pattern_frames[SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE]
            members = pattern_frames[SUPPORT_RESOLUTION_PATTERN_MEMBER_TABLE]
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.iloc[0]["bmu_family_key"], "HOWAO")
            self.assertEqual(int(summary.iloc[0]["open_case_count"]), 2)
            self.assertEqual(int(summary.iloc[0]["open_day_count"]), 2)
            self.assertEqual(summary.iloc[0]["pattern_review_state"], "multi_day")
            self.assertEqual(members["bmu_family_key"].tolist(), ["HOWAO", "HOWAO"])

    def test_annotate_support_resolution_pattern_updates_all_matching_open_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            support_case_family = self._support_case_pattern_frame()
            upsert_frame_to_sqlite(
                db_path,
                "fact_support_case_family_daily",
                support_case_family,
                ["support_batch_id", "settlement_date", "bmu_family_key"],
            )
            materialize_truth_store_support_resolution(
                db_path=db_path,
                generated_at_utc="2026-03-11T12:00:00Z",
            )
            pattern_key = read_support_resolution_pattern_review(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-01_2024-10-07",
                pattern_filter="multi_day",
            )[SUPPORT_RESOLUTION_PATTERN_SUMMARY_TABLE].iloc[0]["support_resolution_pattern_key"]

            annotated = annotate_support_resolution_pattern(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-01_2024-10-07",
                resolution_pattern_key=pattern_key,
                resolution_state="confirmed_publication_gap",
                truth_policy_action="fix_source_and_rerun",
                resolution_note="Bulk-reviewed repeated HOWAO publication-gap pattern.",
                source_reference="bulk-ticket-1",
                generated_at_utc="2026-03-11T13:00:00Z",
            )

            self.assertEqual(annotated["bmu_family_key"].tolist(), ["HOWAO", "HOWAO"])
            refreshed = read_support_case_resolution(
                db_path=db_path,
                support_batch_id="support_fail_warn_days7_families5_2024-10-01_2024-10-07",
                resolution_filter="all",
            )
            resolved_howao = refreshed[refreshed["bmu_family_key"] == "HOWAO"].copy()
            open_moweo = refreshed[refreshed["bmu_family_key"] == "MOWEO"].copy()
            self.assertTrue((resolved_howao["resolution_state"] == "confirmed_publication_gap").all())
            self.assertTrue((resolved_howao["truth_policy_action"] == "fix_source_and_rerun").all())
            self.assertEqual(open_moweo.iloc[0]["resolution_state"], "open")


if __name__ == "__main__":
    unittest.main()
