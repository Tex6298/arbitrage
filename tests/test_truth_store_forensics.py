import unittest

import pandas as pd

from truth_store_forensics import (
    build_fact_family_dispatch_forensic_bmu_daily,
    build_fact_family_dispatch_forensic_daily,
    build_fact_family_dispatch_forensic_half_hourly,
    build_fact_family_physical_forensic_bmu_daily,
    build_fact_family_physical_forensic_daily,
    build_fact_family_physical_forensic_half_hourly,
    build_fact_family_publication_audit_bmu_daily,
    build_fact_family_publication_audit_daily,
    build_fact_family_support_evidence_half_hourly,
)


class TruthStoreForensicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_gap_family = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "bmu_family_key": "HOWAO",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "source_gap_next_action": "add_dispatch_source",
                    "family_source_gap_next_action": "add_dispatch_source",
                    "source_gap_share_of_day_total": 0.4,
                    "source_gap_share_of_remaining_qa_shortfall": 0.25,
                },
                {
                    "settlement_date": "2024-10-02",
                    "bmu_family_key": "HOWBO",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "source_gap_next_action": "add_dispatch_source",
                    "family_source_gap_next_action": "add_dispatch_source",
                    "source_gap_share_of_day_total": 0.35,
                    "source_gap_share_of_remaining_qa_shortfall": 0.20,
                },
            ]
        )
        self.truth = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "interval_start_utc": "2024-10-02T00:00:00Z",
                    "interval_end_utc": "2024-10-02T00:30:00Z",
                    "elexon_bm_unit": "T_HOWAO-1",
                    "national_grid_bm_unit": "T_HOWAO-1",
                    "bmu_family_key": "HOWAO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "negative_bid_available_flag": True,
                    "dispatch_truth_flag": False,
                    "lost_energy_estimate_flag": False,
                    "source_gap_candidate_flag": True,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_evidence_mwh_lower_bound": 0.0,
                    "same_bmu_dispatch_increment_mwh_lower_bound": 0.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    "source_gap_candidate_mwh_lower_bound": 100.0,
                    "acceptance_window_candidate_mwh_lower_bound": 0.0,
                    "family_window_candidate_mwh_lower_bound": 0.0,
                    "no_window_candidate_mwh_lower_bound": 100.0,
                    "physical_dispatch_down_gap_mwh": 100.0,
                    "lost_energy_mwh": 0.0,
                    "availability_state": "available",
                    "lost_energy_block_reason": "no_dispatch_truth",
                    "dispatch_truth_source_tier": "none",
                    "dispatch_inference_scope": "none",
                    "most_negative_bid_gbp_per_mwh": -85.0,
                    "negative_bid_pair_count": 2,
                },
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 2,
                    "interval_start_utc": "2024-10-02T00:30:00Z",
                    "interval_end_utc": "2024-10-02T01:00:00Z",
                    "elexon_bm_unit": "T_HOWAO-2",
                    "national_grid_bm_unit": "T_HOWAO-2",
                    "bmu_family_key": "HOWAO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "negative_bid_available_flag": True,
                    "dispatch_truth_flag": True,
                    "lost_energy_estimate_flag": True,
                    "source_gap_candidate_flag": False,
                    "accepted_down_delta_mwh_lower_bound": 12.0,
                    "dispatch_down_evidence_mwh_lower_bound": 12.0,
                    "same_bmu_dispatch_increment_mwh_lower_bound": 0.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    "source_gap_candidate_mwh_lower_bound": 0.0,
                    "acceptance_window_candidate_mwh_lower_bound": 0.0,
                    "family_window_candidate_mwh_lower_bound": 0.0,
                    "no_window_candidate_mwh_lower_bound": 0.0,
                    "physical_dispatch_down_gap_mwh": 12.0,
                    "lost_energy_mwh": 8.0,
                    "availability_state": "available",
                    "lost_energy_block_reason": "estimated",
                    "dispatch_truth_source_tier": "acceptance_only",
                    "dispatch_inference_scope": "none",
                    "most_negative_bid_gbp_per_mwh": -60.0,
                    "negative_bid_pair_count": 1,
                },
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "interval_start_utc": "2024-10-02T00:00:00Z",
                    "interval_end_utc": "2024-10-02T00:30:00Z",
                    "elexon_bm_unit": "T_HOWBO-1",
                    "national_grid_bm_unit": "T_HOWBO-1",
                    "bmu_family_key": "HOWBO",
                    "bmu_family_label": "Hornsea",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "mapping_status": "mapped",
                    "negative_bid_available_flag": True,
                    "dispatch_truth_flag": False,
                    "lost_energy_estimate_flag": False,
                    "source_gap_candidate_flag": True,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_evidence_mwh_lower_bound": 0.0,
                    "same_bmu_dispatch_increment_mwh_lower_bound": 0.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    "source_gap_candidate_mwh_lower_bound": 80.0,
                    "acceptance_window_candidate_mwh_lower_bound": 0.0,
                    "family_window_candidate_mwh_lower_bound": 0.0,
                    "no_window_candidate_mwh_lower_bound": 80.0,
                    "physical_dispatch_down_gap_mwh": 80.0,
                    "lost_energy_mwh": 0.0,
                    "availability_state": "unknown",
                    "lost_energy_block_reason": "no_dispatch_truth",
                    "dispatch_truth_source_tier": "none",
                    "dispatch_inference_scope": "none",
                    "most_negative_bid_gbp_per_mwh": -90.0,
                    "negative_bid_pair_count": 2,
                },
            ]
        )

    def test_build_family_dispatch_forensic_daily_summarizes_scope(self) -> None:
        frame = build_fact_family_dispatch_forensic_daily(
            self.truth,
            self.source_gap_family,
            family_keys=["HOWAO", "HOWBO"],
        )
        self.assertEqual(len(frame), 2)
        howao = frame[frame["bmu_family_key"] == "HOWAO"].iloc[0]
        self.assertEqual(howao["forensic_state"], "mapped_no_window_source_gap")
        self.assertEqual(int(howao["distinct_bmu_count"]), 2)
        self.assertAlmostEqual(float(howao["source_gap_candidate_mwh_lower_bound"]), 100.0)
        self.assertAlmostEqual(float(howao["dispatch_down_mwh_lower_bound"]), 12.0)
        self.assertEqual(int(howao["negative_bid_half_hour_count"]), 2)

    def test_build_family_dispatch_forensic_bmu_daily_ranks_bmus(self) -> None:
        frame = build_fact_family_dispatch_forensic_bmu_daily(
            self.truth,
            self.source_gap_family,
            family_keys=["HOWAO"],
        )
        self.assertEqual(len(frame), 2)
        first = frame.iloc[0]
        second = frame.iloc[1]
        self.assertEqual(first["elexon_bm_unit"], "T_HOWAO-1")
        self.assertEqual(int(first["bmu_forensic_rank_within_family_day"]), 1)
        self.assertEqual(first["bmu_forensic_state"], "mapped_no_window_source_gap")
        self.assertEqual(second["elexon_bm_unit"], "T_HOWAO-2")
        self.assertEqual(int(second["bmu_forensic_rank_within_family_day"]), 2)

    def test_build_family_dispatch_forensic_half_hourly_labels_candidate_rows(self) -> None:
        frame = build_fact_family_dispatch_forensic_half_hourly(
            self.truth,
            self.source_gap_family,
            family_keys=["HOWAO"],
        )
        self.assertEqual(len(frame), 2)
        candidate = frame[frame["source_gap_candidate_flag"]].iloc[0]
        captured = frame[frame["dispatch_truth_flag"]].iloc[0]
        self.assertEqual(candidate["half_hour_forensic_state"], "no_window_source_gap")
        self.assertEqual(captured["half_hour_forensic_state"], "captured_dispatch")

    def test_build_family_physical_forensic_daily_detects_positive_zero_boalf_gap(self) -> None:
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_HOWAO-1",
                    "pn_mwh": 20.0,
                    "qpn_mwh": 5.0,
                    "mils_mwh": 4.0,
                    "mels_mwh": 22.0,
                    "generation_mwh": 5.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 20.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                },
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_HOWBO-1",
                    "pn_mwh": 18.0,
                    "qpn_mwh": 4.0,
                    "mils_mwh": 3.0,
                    "mels_mwh": 19.0,
                    "generation_mwh": 4.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 18.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                },
            ]
        )

        truth = self.truth.copy()
        truth["sentinel_pair_available_flag"] = False
        frame = build_fact_family_physical_forensic_daily(
            physical,
            truth,
            self.source_gap_family,
            family_keys=["HOWAO", "HOWBO"],
        )
        howao = frame[frame["bmu_family_key"] == "HOWAO"].iloc[0]
        self.assertEqual(howao["physical_forensic_state"], "positive_zero_boalf_negative_bid_gap")
        self.assertEqual(int(howao["positive_zero_boalf_gap_row_count"]), 1)
        self.assertEqual(int(howao["positive_zero_boalf_negative_bid_gap_row_count"]), 1)

    def test_build_family_physical_forensic_bmu_daily_flags_sentinel_gap(self) -> None:
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_HOWBO-1",
                    "pn_mwh": 18.0,
                    "qpn_mwh": 4.0,
                    "mils_mwh": 3.0,
                    "mels_mwh": 19.0,
                    "generation_mwh": 4.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 18.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                }
            ]
        )
        truth = self.truth[self.truth["elexon_bm_unit"] == "T_HOWBO-1"].copy()
        truth["sentinel_pair_available_flag"] = True
        frame = build_fact_family_physical_forensic_bmu_daily(
            physical,
            truth,
            self.source_gap_family,
            family_keys=["HOWBO"],
        )
        first = frame.iloc[0]
        self.assertEqual(first["bmu_physical_forensic_state"], "positive_zero_boalf_negative_bid_gap")
        self.assertEqual(int(first["positive_zero_boalf_sentinel_gap_row_count"]), 1)

    def test_build_family_physical_forensic_half_hourly_labels_positive_zero_boalf_gap(self) -> None:
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_HOWAO-1",
                    "pn_mwh": 20.0,
                    "qpn_mwh": 5.0,
                    "mils_mwh": 4.0,
                    "mels_mwh": 22.0,
                    "generation_mwh": 5.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 20.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                },
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_HOWAO-2",
                    "pn_mwh": 12.0,
                    "qpn_mwh": 12.0,
                    "mils_mwh": 10.0,
                    "mels_mwh": 14.0,
                    "generation_mwh": 10.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 12.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                },
            ]
        )
        truth = self.truth[self.truth["bmu_family_key"] == "HOWAO"].copy()
        truth["sentinel_pair_available_flag"] = False
        frame = build_fact_family_physical_forensic_half_hourly(
            physical,
            truth,
            self.source_gap_family,
            family_keys=["HOWAO"],
        )
        gap_row = frame[frame["elexon_bm_unit"] == "T_HOWAO-1"].iloc[0]
        captured_row = frame[frame["elexon_bm_unit"] == "T_HOWAO-2"].iloc[0]
        self.assertEqual(gap_row["physical_half_hour_forensic_state"], "positive_zero_boalf_negative_bid_gap")
        self.assertEqual(captured_row["physical_half_hour_forensic_state"], "captured_dispatch_present")

    def test_build_family_publication_audit_daily_marks_support_case(self) -> None:
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_HOWAO-1",
                    "pn_mwh": 20.0,
                    "qpn_mwh": 5.0,
                    "mils_mwh": 4.0,
                    "mels_mwh": 22.0,
                    "generation_mwh": 5.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 20.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                }
            ]
        )
        truth = self.truth[self.truth["elexon_bm_unit"] == "T_HOWAO-1"].copy()
        truth["sentinel_pair_available_flag"] = False
        frame = build_fact_family_publication_audit_daily(
            physical,
            truth,
            self.source_gap_family,
            family_keys=["HOWAO"],
        )
        first = frame.iloc[0]
        self.assertEqual(first["publication_audit_state"], "physical_without_boalf_negative_bid")
        self.assertEqual(first["support_question_code"], "query_missing_boalf_with_negative_bid_and_physical_gap")
        self.assertEqual(int(first["physical_without_boalf_half_hour_count"]), 1)

    def test_build_family_publication_audit_bmu_daily_detects_dynamic_limit_case(self) -> None:
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_HOWBO-1",
                    "pn_mwh": 18.0,
                    "qpn_mwh": 4.0,
                    "mils_mwh": 3.0,
                    "mels_mwh": 9.0,
                    "generation_mwh": 4.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 18.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                }
            ]
        )
        truth = self.truth[self.truth["elexon_bm_unit"] == "T_HOWBO-1"].copy()
        truth["negative_bid_available_flag"] = False
        truth["sentinel_pair_available_flag"] = False
        frame = build_fact_family_publication_audit_bmu_daily(
            physical,
            truth,
            self.source_gap_family,
            family_keys=["HOWBO"],
        )
        first = frame.iloc[0]
        self.assertEqual(first["bmu_publication_audit_state"], "availability_like_dynamic_limit")
        self.assertEqual(first["support_question_code"], "query_dynamic_limit_change_without_boalf")

    def test_build_family_support_evidence_half_hourly_creates_support_case_key(self) -> None:
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_HOWBO-1",
                    "pn_mwh": 18.0,
                    "qpn_mwh": 4.0,
                    "mils_mwh": 3.0,
                    "mels_mwh": 19.0,
                    "generation_mwh": 4.0,
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 18.0,
                    "physical_consistency_flag": True,
                    "counterfactual_valid_flag": True,
                }
            ]
        )
        truth = self.truth[self.truth["elexon_bm_unit"] == "T_HOWBO-1"].copy()
        truth["sentinel_bid_pair_count"] = 1
        truth["sentinel_offer_pair_count"] = 6
        truth["sentinel_pair_count"] = 7
        truth["sentinel_pair_available_flag"] = True
        frame = build_fact_family_support_evidence_half_hourly(
            physical,
            truth,
            self.source_gap_family,
            family_keys=["HOWBO"],
        )
        first = frame.iloc[0]
        self.assertEqual(first["publication_audit_state"], "physical_without_boalf_sentinel_bod_present")
        self.assertTrue(str(first["support_case_key"]).startswith("HOWBO:2024-10-02:T_HOWBO-1:SP1"))
        self.assertEqual(first["support_question_code"], "query_bod_sentinel_and_missing_boalf")


if __name__ == "__main__":
    unittest.main()
