import tempfile
import unittest
from pathlib import Path

import pandas as pd

from history_store import upsert_frame_to_sqlite
from support_loop import (
    SUPPORT_CASE_DAILY_TABLE,
    SUPPORT_CASE_FAMILY_TABLE,
    SUPPORT_CASE_HALF_HOURLY_TABLE,
    SUPPORT_SUMMARY_FILENAME,
    build_support_case_summary_markdown,
    materialize_truth_store_support_loop,
    select_support_case_daily,
    select_support_case_family_daily,
)


class SupportLoopSelectionTests(unittest.TestCase):
    def test_select_support_case_daily_and_family_respects_support_query_actions(self) -> None:
        anomaly_daily = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "remaining_qa_shortfall_mwh": 100.0,
                    "publication_anomaly_family_count": 2,
                    "publication_anomaly_row_count": 5,
                    "publication_anomaly_distinct_bmu_count": 2,
                    "publication_anomaly_candidate_mwh_lower_bound": 50.0,
                    "publication_anomaly_negative_bid_mwh_lower_bound": 0.0,
                    "publication_anomaly_sentinel_mwh_lower_bound": 50.0,
                    "publication_anomaly_dynamic_limit_mwh_lower_bound": 0.0,
                    "publication_anomaly_other_mwh_lower_bound": 0.0,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.5,
                    "publication_anomaly_dominant_state": "sentinel_bod_present",
                    "publication_anomaly_priority_rank": 1,
                    "publication_anomaly_next_action": "support_query_bod_sentinel_and_boalf_publication",
                },
                {
                    "settlement_date": "2024-10-03",
                    "qa_reconciliation_status": "warn",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "remaining_qa_shortfall_mwh": 80.0,
                    "publication_anomaly_family_count": 1,
                    "publication_anomaly_row_count": 3,
                    "publication_anomaly_distinct_bmu_count": 1,
                    "publication_anomaly_candidate_mwh_lower_bound": 30.0,
                    "publication_anomaly_negative_bid_mwh_lower_bound": 30.0,
                    "publication_anomaly_sentinel_mwh_lower_bound": 0.0,
                    "publication_anomaly_dynamic_limit_mwh_lower_bound": 0.0,
                    "publication_anomaly_other_mwh_lower_bound": 0.0,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.375,
                    "publication_anomaly_dominant_state": "negative_bid_without_boalf",
                    "publication_anomaly_priority_rank": 2,
                    "publication_anomaly_next_action": "support_query_missing_published_boalf",
                },
                {
                    "settlement_date": "2024-10-04",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "remaining_qa_shortfall_mwh": 70.0,
                    "publication_anomaly_family_count": 1,
                    "publication_anomaly_row_count": 2,
                    "publication_anomaly_distinct_bmu_count": 1,
                    "publication_anomaly_candidate_mwh_lower_bound": 20.0,
                    "publication_anomaly_negative_bid_mwh_lower_bound": 0.0,
                    "publication_anomaly_sentinel_mwh_lower_bound": 0.0,
                    "publication_anomaly_dynamic_limit_mwh_lower_bound": 20.0,
                    "publication_anomaly_other_mwh_lower_bound": 0.0,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.286,
                    "publication_anomaly_dominant_state": "dynamic_limit_like_without_boalf",
                    "publication_anomaly_priority_rank": 3,
                    "publication_anomaly_next_action": "inspect_dynamic_limit_publication",
                },
            ]
        )
        anomaly_family = pd.DataFrame(
            [
                {
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
                    "publication_anomaly_row_count": 3,
                    "publication_anomaly_distinct_bmu_count": 1,
                    "publication_anomaly_candidate_mwh_lower_bound": 30.0,
                    "publication_anomaly_negative_bid_mwh_lower_bound": 0.0,
                    "publication_anomaly_sentinel_mwh_lower_bound": 30.0,
                    "publication_anomaly_dynamic_limit_mwh_lower_bound": 0.0,
                    "publication_anomaly_other_mwh_lower_bound": 0.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_mwh_lower_bound": 0.0,
                    "lost_energy_mwh": 0.0,
                    "publication_anomaly_share_of_day_total": 0.6,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.3,
                    "publication_anomaly_dominant_state": "sentinel_bod_present",
                    "day_family_rank_by_publication_anomaly": 1,
                    "family_publication_anomaly_next_action": "support_query_bod_sentinel_and_boalf_publication",
                },
                {
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "publication_anomaly_next_action": "support_query_bod_sentinel_and_boalf_publication",
                    "bmu_family_key": "WLNYO",
                    "bmu_family_label": "Walney",
                    "cluster_key": pd.NA,
                    "cluster_label": pd.NA,
                    "parent_region": pd.NA,
                    "mapping_status": "unmapped",
                    "publication_anomaly_row_count": 2,
                    "publication_anomaly_distinct_bmu_count": 1,
                    "publication_anomaly_candidate_mwh_lower_bound": 10.0,
                    "publication_anomaly_negative_bid_mwh_lower_bound": 10.0,
                    "publication_anomaly_sentinel_mwh_lower_bound": 0.0,
                    "publication_anomaly_dynamic_limit_mwh_lower_bound": 0.0,
                    "publication_anomaly_other_mwh_lower_bound": 0.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_mwh_lower_bound": 0.0,
                    "lost_energy_mwh": 0.0,
                    "publication_anomaly_share_of_day_total": 0.2,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.1,
                    "publication_anomaly_dominant_state": "negative_bid_without_boalf",
                    "day_family_rank_by_publication_anomaly": 2,
                    "family_publication_anomaly_next_action": "mapping_and_publication_audit",
                },
                {
                    "settlement_date": "2024-10-03",
                    "qa_reconciliation_status": "warn",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "publication_anomaly_next_action": "support_query_missing_published_boalf",
                    "bmu_family_key": "HOWAO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "publication_anomaly_row_count": 3,
                    "publication_anomaly_distinct_bmu_count": 1,
                    "publication_anomaly_candidate_mwh_lower_bound": 30.0,
                    "publication_anomaly_negative_bid_mwh_lower_bound": 30.0,
                    "publication_anomaly_sentinel_mwh_lower_bound": 0.0,
                    "publication_anomaly_dynamic_limit_mwh_lower_bound": 0.0,
                    "publication_anomaly_other_mwh_lower_bound": 0.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_mwh_lower_bound": 0.0,
                    "lost_energy_mwh": 0.0,
                    "publication_anomaly_share_of_day_total": 1.0,
                    "publication_anomaly_share_of_remaining_qa_shortfall": 0.375,
                    "publication_anomaly_dominant_state": "negative_bid_without_boalf",
                    "day_family_rank_by_publication_anomaly": 1,
                    "family_publication_anomaly_next_action": "support_query_missing_published_boalf",
                },
            ]
        )

        selected_days = select_support_case_daily(anomaly_daily, status_mode="fail_warn", top_days=5)
        self.assertEqual(selected_days["settlement_date"].tolist(), ["2024-10-02", "2024-10-03"])
        self.assertEqual(selected_days["support_case_day_rank"].tolist(), [1, 2])

        selected_families = select_support_case_family_daily(
            anomaly_family,
            selected_support_days=selected_days,
            top_families_per_day=5,
        )
        self.assertEqual(selected_families["bmu_family_key"].tolist(), ["HOWBO", "HOWAO"])
        self.assertEqual(selected_families["support_case_family_rank"].tolist(), [1, 1])


class SupportLoopMarkdownTests(unittest.TestCase):
    def test_build_support_case_summary_markdown_contains_expected_sections(self) -> None:
        daily = pd.DataFrame(
            [
                {
                    "support_case_day_rank": 1,
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "publication_anomaly_dominant_state": "sentinel_bod_present",
                    "publication_anomaly_candidate_mwh_lower_bound": 50.0,
                    "remaining_qa_shortfall_mwh": 100.0,
                    "publication_anomaly_next_action": "support_query_bod_sentinel_and_boalf_publication",
                    "support_recommended_action": "ask_elexon_about_suspect_bod_sentinel_and_missing_published_boalf",
                    "selected_family_count": 1,
                }
            ]
        )
        family = pd.DataFrame(
            [
                {
                    "support_case_family_rank": 1,
                    "settlement_date": "2024-10-02",
                    "bmu_family_key": "HOWBO",
                    "bmu_family_label": "Hornsea",
                    "publication_anomaly_dominant_state": "sentinel_bod_present",
                    "publication_anomaly_candidate_mwh_lower_bound": 30.0,
                    "parent_region": "England/Wales",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "mapping_status": "mapped",
                    "support_question_code": "query_bod_sentinel_and_missing_boalf",
                    "support_recommended_action": "ask_elexon_about_suspect_bod_sentinel_and_missing_published_boalf",
                }
            ]
        )
        half_hourly = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "bmu_family_key": "HOWBO",
                    "elexon_bm_unit": "T_HOWBO-1",
                    "published_boalf_absent_flag": True,
                    "negative_bid_available_flag": False,
                    "sentinel_pair_available_flag": True,
                    "publication_audit_state": "physical_without_boalf_sentinel_bod_present",
                    "physical_dispatch_down_gap_mwh": 30.0,
                    "settlement_period": 1,
                    "interval_start_utc": "2024-10-02T00:00:00Z",
                    "most_negative_bid_gbp_per_mwh": -9999.0,
                    "sentinel_pair_count": 7,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                }
            ]
        )

        markdown = build_support_case_summary_markdown(
            fact_support_case_daily=daily,
            fact_support_case_family_daily=family,
            fact_support_case_half_hourly=half_hourly,
            source_db_path="bmu_truth_store.sqlite",
            support_batch_id="support_fail_warn_days7_families5_2024-10-02_2024-10-02",
            support_status_mode="fail_warn",
            support_top_days=7,
            support_top_families_per_day=5,
            support_generated_at_utc="2026-03-11T12:00:00Z",
            example_half_hour_limit=10,
        )

        self.assertIn("# Support Case Summary", markdown)
        self.assertIn("Day 1: 2024-10-02", markdown)
        self.assertIn("query_bod_sentinel_and_missing_boalf", markdown)
        self.assertIn("T_HOWBO-1", markdown)
        self.assertIn("sentinel_pair_count", markdown)


class SupportLoopMaterializationTests(unittest.TestCase):
    def test_materialize_truth_store_support_loop_writes_store_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "truth_store.sqlite"
            output_dir = Path(temp_dir) / "support_output"

            reconciliation_daily = pd.DataFrame(
                [
                    {
                        "settlement_date": "2024-10-02",
                        "qa_reconciliation_status": "fail",
                        "qa_target_definition": "wind_constraints_positive_only_v1",
                        "gb_daily_qa_target_mwh": 100.0,
                        "gb_daily_estimated_lost_energy_mwh": 0.0,
                        "lost_energy_capture_ratio_vs_qa_target": 0.0,
                        "dispatch_half_hour_count": 0,
                        "dispatch_family_day_inference_row_count": 0,
                        "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "qa_reconciliation_status": "fail",
                        "qa_target_definition": "wind_constraints_positive_only_v1",
                        "gb_daily_qa_target_mwh": 80.0,
                        "gb_daily_estimated_lost_energy_mwh": 0.0,
                        "lost_energy_capture_ratio_vs_qa_target": 0.0,
                        "dispatch_half_hour_count": 0,
                        "dispatch_family_day_inference_row_count": 0,
                        "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    },
                ]
            )
            target_audit_daily = pd.DataFrame(
                [
                    {
                        "settlement_date": "2024-10-02",
                        "qa_target_definition": "wind_constraints_positive_only_v1",
                        "recoverability_audit_state": "source_limited",
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "qa_target_definition": "wind_constraints_positive_only_v1",
                        "recoverability_audit_state": "source_limited",
                    },
                ]
            )
            family_shortfall_daily = pd.DataFrame(
                [
                    {
                        "settlement_date": "2024-10-02",
                        "bmu_family_key": "HOWBO",
                        "bmu_family_label": "Hornsea",
                        "cluster_key": "dogger_hornsea_offshore",
                        "cluster_label": "Dogger and Hornsea Offshore",
                        "parent_region": "England/Wales",
                        "mapping_status": "mapped",
                        "dispatch_half_hour_count": 0,
                        "lost_energy_estimate_half_hour_count": 0,
                        "dispatch_minus_lost_energy_gap_mwh": 50.0,
                        "share_of_day_remaining_qa_shortfall": 0.5,
                        "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "bmu_family_key": "HOWAO",
                        "bmu_family_label": "Hornsea",
                        "cluster_key": "dogger_hornsea_offshore",
                        "cluster_label": "Dogger and Hornsea Offshore",
                        "parent_region": "England/Wales",
                        "mapping_status": "mapped",
                        "dispatch_half_hour_count": 0,
                        "lost_energy_estimate_half_hour_count": 0,
                        "dispatch_minus_lost_energy_gap_mwh": 40.0,
                        "share_of_day_remaining_qa_shortfall": 0.5,
                        "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "bmu_family_key": "WLNYO",
                        "bmu_family_label": "Walney",
                        "cluster_key": pd.NA,
                        "cluster_label": pd.NA,
                        "parent_region": pd.NA,
                        "mapping_status": "unmapped",
                        "dispatch_half_hour_count": 0,
                        "lost_energy_estimate_half_hour_count": 0,
                        "dispatch_minus_lost_energy_gap_mwh": 25.0,
                        "share_of_day_remaining_qa_shortfall": 0.3125,
                        "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    },
                ]
            )
            truth_half_hourly = pd.DataFrame(
                [
                    {
                        "settlement_date": "2024-10-02",
                        "settlement_period": 1,
                        "interval_start_utc": "2024-10-02T00:00:00Z",
                        "interval_end_utc": "2024-10-02T00:30:00Z",
                        "elexon_bm_unit": "T_HOWBO-1",
                        "national_grid_bm_unit": "HOWBO1",
                        "bmu_family_key": "HOWBO",
                        "bmu_family_label": "Hornsea",
                        "cluster_key": "dogger_hornsea_offshore",
                        "cluster_label": "Dogger and Hornsea Offshore",
                        "parent_region": "England/Wales",
                        "mapping_status": "mapped",
                        "accepted_down_delta_mwh_lower_bound": 0.0,
                        "dispatch_down_evidence_mwh_lower_bound": 0.0,
                        "physical_dispatch_down_gap_mwh": 50.0,
                        "negative_bid_available_flag": False,
                        "negative_bid_pair_count": 0,
                        "valid_negative_bid_pair_count": 0,
                        "sentinel_bid_pair_count": 1,
                        "sentinel_offer_pair_count": 6,
                        "sentinel_pair_count": 7,
                        "sentinel_pair_available_flag": True,
                        "most_negative_bid_gbp_per_mwh": -9999.0,
                        "dispatch_truth_flag": False,
                        "lost_energy_estimate_flag": False,
                        "lost_energy_mwh": 0.0,
                        "lost_energy_block_reason": "no_dispatch_truth",
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "settlement_period": 1,
                        "interval_start_utc": "2024-10-06T00:00:00Z",
                        "interval_end_utc": "2024-10-06T00:30:00Z",
                        "elexon_bm_unit": "T_HOWAO-1",
                        "national_grid_bm_unit": "HOWAO1",
                        "bmu_family_key": "HOWAO",
                        "bmu_family_label": "Hornsea",
                        "cluster_key": "dogger_hornsea_offshore",
                        "cluster_label": "Dogger and Hornsea Offshore",
                        "parent_region": "England/Wales",
                        "mapping_status": "mapped",
                        "accepted_down_delta_mwh_lower_bound": 0.0,
                        "dispatch_down_evidence_mwh_lower_bound": 0.0,
                        "physical_dispatch_down_gap_mwh": 40.0,
                        "negative_bid_available_flag": True,
                        "negative_bid_pair_count": 1,
                        "valid_negative_bid_pair_count": 1,
                        "sentinel_bid_pair_count": 0,
                        "sentinel_offer_pair_count": 0,
                        "sentinel_pair_count": 0,
                        "sentinel_pair_available_flag": False,
                        "most_negative_bid_gbp_per_mwh": -75.0,
                        "dispatch_truth_flag": False,
                        "lost_energy_estimate_flag": False,
                        "lost_energy_mwh": 0.0,
                        "lost_energy_block_reason": "no_dispatch_truth",
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "settlement_period": 2,
                        "interval_start_utc": "2024-10-06T00:30:00Z",
                        "interval_end_utc": "2024-10-06T01:00:00Z",
                        "elexon_bm_unit": "T_WLNYO-1",
                        "national_grid_bm_unit": "WLNYO1",
                        "bmu_family_key": "WLNYO",
                        "bmu_family_label": "Walney",
                        "cluster_key": pd.NA,
                        "cluster_label": pd.NA,
                        "parent_region": pd.NA,
                        "mapping_status": "unmapped",
                        "accepted_down_delta_mwh_lower_bound": 0.0,
                        "dispatch_down_evidence_mwh_lower_bound": 0.0,
                        "physical_dispatch_down_gap_mwh": 25.0,
                        "negative_bid_available_flag": True,
                        "negative_bid_pair_count": 1,
                        "valid_negative_bid_pair_count": 1,
                        "sentinel_bid_pair_count": 0,
                        "sentinel_offer_pair_count": 0,
                        "sentinel_pair_count": 0,
                        "sentinel_pair_available_flag": False,
                        "most_negative_bid_gbp_per_mwh": -70.0,
                        "dispatch_truth_flag": False,
                        "lost_energy_estimate_flag": False,
                        "lost_energy_mwh": 0.0,
                        "lost_energy_block_reason": "no_dispatch_truth",
                    },
                ]
            )
            physical_half_hourly = pd.DataFrame(
                [
                    {
                        "settlement_date": "2024-10-02",
                        "settlement_period": 1,
                        "elexon_bm_unit": "T_HOWBO-1",
                        "pn_mwh": 50.0,
                        "qpn_mwh": 0.0,
                        "mils_mwh": 0.0,
                        "mels_mwh": 52.0,
                        "generation_mwh": 0.0,
                        "physical_baseline_source_dataset": "PN",
                        "physical_baseline_mwh": 50.0,
                        "physical_consistency_flag": True,
                        "counterfactual_valid_flag": True,
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "settlement_period": 1,
                        "elexon_bm_unit": "T_HOWAO-1",
                        "pn_mwh": 40.0,
                        "qpn_mwh": 0.0,
                        "mils_mwh": 0.0,
                        "mels_mwh": 42.0,
                        "generation_mwh": 0.0,
                        "physical_baseline_source_dataset": "PN",
                        "physical_baseline_mwh": 40.0,
                        "physical_consistency_flag": True,
                        "counterfactual_valid_flag": True,
                    },
                    {
                        "settlement_date": "2024-10-06",
                        "settlement_period": 2,
                        "elexon_bm_unit": "T_WLNYO-1",
                        "pn_mwh": 25.0,
                        "qpn_mwh": 0.0,
                        "mils_mwh": 0.0,
                        "mels_mwh": 26.0,
                        "generation_mwh": 0.0,
                        "physical_baseline_source_dataset": "PN",
                        "physical_baseline_mwh": 25.0,
                        "physical_consistency_flag": True,
                        "counterfactual_valid_flag": True,
                    },
                ]
            )

            upsert_frame_to_sqlite(db_path, "fact_curtailment_reconciliation_daily", reconciliation_daily, ["settlement_date"])
            upsert_frame_to_sqlite(db_path, "fact_constraint_target_audit_daily", target_audit_daily, ["settlement_date"])
            upsert_frame_to_sqlite(
                db_path,
                "fact_bmu_family_shortfall_daily",
                family_shortfall_daily,
                ["settlement_date", "bmu_family_key"],
            )
            upsert_frame_to_sqlite(
                db_path,
                "fact_bmu_curtailment_truth_half_hourly",
                truth_half_hourly,
                ["settlement_date", "settlement_period", "elexon_bm_unit"],
            )
            upsert_frame_to_sqlite(
                db_path,
                "fact_bmu_physical_position_half_hourly",
                physical_half_hourly,
                ["settlement_date", "settlement_period", "elexon_bm_unit"],
            )

            support_batch_id, frames, summary_markdown, written_paths = materialize_truth_store_support_loop(
                db_path=db_path,
                status_mode="fail_warn",
                top_days=7,
                top_families_per_day=5,
                output_dir=output_dir,
                generated_at_utc="2026-03-11T12:00:00Z",
            )

            self.assertEqual(support_batch_id, "support_fail_warn_days7_families5_2024-10-02_2024-10-06")
            self.assertEqual(len(frames[SUPPORT_CASE_DAILY_TABLE]), 2)
            self.assertCountEqual(
                frames[SUPPORT_CASE_FAMILY_TABLE]["bmu_family_key"].tolist(),
                ["HOWBO", "HOWAO"],
            )
            self.assertNotIn("WLNYO", frames[SUPPORT_CASE_FAMILY_TABLE]["bmu_family_key"].tolist())
            self.assertCountEqual(
                frames[SUPPORT_CASE_HALF_HOURLY_TABLE]["elexon_bm_unit"].tolist(),
                ["T_HOWBO-1", "T_HOWAO-1"],
            )
            self.assertIn("query_bod_sentinel_and_missing_boalf", summary_markdown)
            self.assertIn("query_missing_boalf_with_negative_bid_and_physical_gap", summary_markdown)

            self.assertTrue(written_paths[SUPPORT_CASE_DAILY_TABLE].exists())
            self.assertTrue(written_paths[SUPPORT_CASE_FAMILY_TABLE].exists())
            self.assertTrue(written_paths[SUPPORT_CASE_HALF_HOURLY_TABLE].exists())
            self.assertTrue(written_paths[SUPPORT_SUMMARY_FILENAME].exists())


if __name__ == "__main__":
    unittest.main()
