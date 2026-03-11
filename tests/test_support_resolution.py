import tempfile
import unittest
from pathlib import Path

import pandas as pd

from history_store import upsert_frame_to_sqlite
from support_resolution import (
    SUPPORT_CASE_RESOLUTION_TABLE,
    SUPPORT_RESOLUTION_BATCH_TABLE,
    SUPPORT_RESOLUTION_DAILY_TABLE,
    annotate_support_case_resolution,
    materialize_truth_store_support_resolution,
    read_support_case_resolution,
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


if __name__ == "__main__":
    unittest.main()
