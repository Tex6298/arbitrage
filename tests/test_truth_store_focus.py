import unittest

import pandas as pd

from truth_store_focus import (
    build_fact_dispatch_source_gap_daily,
    build_fact_dispatch_source_gap_family_daily,
    build_fact_source_completeness_focus_daily,
    build_fact_source_completeness_focus_family_daily,
)


class TruthStoreFocusTests(unittest.TestCase):
    def test_build_source_completeness_focus_daily_assigns_next_action(self) -> None:
        reconciliation = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "qa_target_definition": "wind_constraints_positive_only_v1",
                    "gb_daily_qa_target_mwh": 100.0,
                    "gb_daily_estimated_lost_energy_mwh": 10.0,
                    "lost_energy_capture_ratio_vs_qa_target": 0.10,
                    "dispatch_half_hour_count": 20,
                    "dispatch_family_day_inference_row_count": 0,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                },
                {
                    "settlement_date": "2024-10-04",
                    "qa_reconciliation_status": "warn",
                    "qa_target_definition": "wind_constraints_positive_only_v1",
                    "gb_daily_qa_target_mwh": 100.0,
                    "gb_daily_estimated_lost_energy_mwh": 70.0,
                    "lost_energy_capture_ratio_vs_qa_target": 0.70,
                    "dispatch_half_hour_count": 200,
                    "dispatch_family_day_inference_row_count": 60,
                    "family_day_dispatch_increment_mwh_lower_bound": 500.0,
                },
            ]
        )
        audit = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "qa_target_definition": "wind_constraints_positive_only_v1",
                    "recoverability_audit_state": "source_limited",
                },
                {
                    "settlement_date": "2024-10-04",
                    "qa_target_definition": "wind_constraints_positive_only_v1",
                    "recoverability_audit_state": "partially_recovered",
                },
            ]
        )

        focus = build_fact_source_completeness_focus_daily(reconciliation, audit)
        self.assertEqual(focus.iloc[0]["settlement_date"], "2024-10-02")
        self.assertEqual(focus.iloc[0]["next_action"], "add_dispatch_source")
        self.assertEqual(focus.iloc[1]["next_action"], "expand_source_after_family_window")
        self.assertEqual(int(focus.iloc[0]["focus_priority_rank"]), 1)

    def test_build_source_completeness_focus_family_daily_ranks_family_gaps(self) -> None:
        daily_focus = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "source_completeness_focus_flag": True,
                },
                {
                    "settlement_date": "2024-10-04",
                    "qa_reconciliation_status": "warn",
                    "recoverability_audit_state": "partially_recovered",
                    "next_action": "expand_source_after_family_window",
                    "source_completeness_focus_flag": True,
                },
            ]
        )
        family = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "bmu_family_key": "SGRWO",
                    "bmu_family_label": "Seagreen",
                    "cluster_key": "east_coast_scotland_offshore",
                    "cluster_label": "East Coast Scotland Offshore",
                    "parent_region": "Scotland",
                    "mapping_status": "mapped",
                    "dispatch_half_hour_count": 5,
                    "lost_energy_estimate_half_hour_count": 1,
                    "dispatch_minus_lost_energy_gap_mwh": 400.0,
                    "share_of_day_remaining_qa_shortfall": 0.3,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                },
                {
                    "settlement_date": "2024-10-04",
                    "bmu_family_key": "MOWEO",
                    "bmu_family_label": "Moray East",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                    "mapping_status": "mapped",
                    "dispatch_half_hour_count": 40,
                    "lost_energy_estimate_half_hour_count": 30,
                    "dispatch_minus_lost_energy_gap_mwh": 1000.0,
                    "share_of_day_remaining_qa_shortfall": 0.6,
                    "family_day_dispatch_increment_mwh_lower_bound": 600.0,
                },
            ]
        )

        focus_family = build_fact_source_completeness_focus_family_daily(family, daily_focus)
        self.assertEqual(len(focus_family), 2)
        self.assertEqual(focus_family.iloc[0]["family_next_action"], "missing_dispatch_evidence")
        self.assertEqual(focus_family.iloc[1]["family_next_action"], "expand_dispatch_source_beyond_family_window")
        self.assertEqual(int(focus_family.iloc[0]["day_family_rank_by_gap"]), 1)
        self.assertEqual(int(focus_family.iloc[1]["day_family_rank_by_gap"]), 1)

    def test_build_dispatch_source_gap_daily_flags_no_window_source_day(self) -> None:
        daily_focus = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "remaining_qa_shortfall_mwh": 100.0,
                },
                {
                    "settlement_date": "2024-10-04",
                    "qa_reconciliation_status": "warn",
                    "recoverability_audit_state": "partially_recovered",
                    "next_action": "expand_source_after_family_window",
                    "remaining_qa_shortfall_mwh": 80.0,
                },
            ]
        )
        truth = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "E_SHRSW-1",
                    "bmu_family_key": "SHRSW",
                    "bmu_family_label": "Sheirds",
                    "cluster_key": pd.NA,
                    "cluster_label": pd.NA,
                    "parent_region": "Scotland",
                    "mapping_status": "unmapped",
                    "dispatch_truth_flag": False,
                    "negative_bid_available_flag": True,
                    "dispatch_acceptance_window_flag": False,
                    "family_day_dispatch_window_flag": False,
                    "family_day_dispatch_expansion_eligible_flag": False,
                    "physical_dispatch_down_gap_mwh": 20.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_evidence_mwh_lower_bound": 0.0,
                    "lost_energy_mwh": 0.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    "lost_energy_block_reason": "no_dispatch_truth",
                },
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 2,
                    "elexon_bm_unit": "E_BTUIW-2",
                    "bmu_family_key": "BTUIW",
                    "bmu_family_label": "Beinn Tharsuinn",
                    "cluster_key": pd.NA,
                    "cluster_label": pd.NA,
                    "parent_region": "Scotland",
                    "mapping_status": "unmapped",
                    "dispatch_truth_flag": False,
                    "negative_bid_available_flag": True,
                    "dispatch_acceptance_window_flag": False,
                    "family_day_dispatch_window_flag": False,
                    "family_day_dispatch_expansion_eligible_flag": False,
                    "physical_dispatch_down_gap_mwh": 10.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_evidence_mwh_lower_bound": 0.0,
                    "lost_energy_mwh": 0.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    "lost_energy_block_reason": "no_dispatch_truth",
                },
                {
                    "settlement_date": "2024-10-04",
                    "settlement_period": 3,
                    "elexon_bm_unit": "T_MOWEO-1",
                    "bmu_family_key": "MOWEO",
                    "bmu_family_label": "Moray East",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                    "mapping_status": "mapped",
                    "dispatch_truth_flag": False,
                    "negative_bid_available_flag": True,
                    "dispatch_acceptance_window_flag": False,
                    "family_day_dispatch_window_flag": True,
                    "family_day_dispatch_expansion_eligible_flag": True,
                    "physical_dispatch_down_gap_mwh": 30.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_evidence_mwh_lower_bound": 0.0,
                    "lost_energy_mwh": 0.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 15.0,
                    "lost_energy_block_reason": "no_dispatch_truth",
                },
            ]
        )

        gap_daily = build_fact_dispatch_source_gap_daily(truth, daily_focus)
        first_day = gap_daily[gap_daily["settlement_date"] == "2024-10-02"].iloc[0]
        second_day = gap_daily[gap_daily["settlement_date"] == "2024-10-04"].iloc[0]

        self.assertAlmostEqual(float(first_day["source_gap_candidate_mwh_lower_bound"]), 30.0)
        self.assertEqual(first_day["source_gap_dominant_scope"], "no_window")
        self.assertEqual(first_day["source_gap_next_action"], "mapping_and_source_audit")
        self.assertAlmostEqual(float(first_day["source_gap_share_of_remaining_qa_shortfall"]), 0.30)

        self.assertAlmostEqual(float(second_day["family_window_candidate_mwh_lower_bound"]), 30.0)
        self.assertEqual(second_day["source_gap_dominant_scope"], "family_window")
        self.assertEqual(second_day["source_gap_next_action"], "inspect_window_rule_thresholds")

    def test_build_dispatch_source_gap_family_daily_ranks_candidates(self) -> None:
        daily_gap = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "qa_reconciliation_status": "fail",
                    "recoverability_audit_state": "source_limited",
                    "next_action": "add_dispatch_source",
                    "remaining_qa_shortfall_mwh": 100.0,
                    "source_gap_next_action": "add_dispatch_source",
                }
            ]
        )
        truth = pd.DataFrame(
            [
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 1,
                    "elexon_bm_unit": "E_SHRSW-1",
                    "bmu_family_key": "SHRSW",
                    "bmu_family_label": "Sheirds",
                    "cluster_key": pd.NA,
                    "cluster_label": pd.NA,
                    "parent_region": "Scotland",
                    "mapping_status": "unmapped",
                    "dispatch_truth_flag": False,
                    "negative_bid_available_flag": True,
                    "dispatch_acceptance_window_flag": False,
                    "family_day_dispatch_window_flag": False,
                    "family_day_dispatch_expansion_eligible_flag": False,
                    "physical_dispatch_down_gap_mwh": 25.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_evidence_mwh_lower_bound": 0.0,
                    "lost_energy_mwh": 0.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 0.0,
                    "lost_energy_block_reason": "no_dispatch_truth",
                },
                {
                    "settlement_date": "2024-10-02",
                    "settlement_period": 2,
                    "elexon_bm_unit": "E_KYPEW-1",
                    "bmu_family_key": "KYPEW",
                    "bmu_family_label": "Kype Muir",
                    "cluster_key": "south_scotland_onshore",
                    "cluster_label": "South Scotland Onshore",
                    "parent_region": "Scotland",
                    "mapping_status": "mapped",
                    "dispatch_truth_flag": False,
                    "negative_bid_available_flag": True,
                    "dispatch_acceptance_window_flag": False,
                    "family_day_dispatch_window_flag": False,
                    "family_day_dispatch_expansion_eligible_flag": True,
                    "physical_dispatch_down_gap_mwh": 15.0,
                    "accepted_down_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_evidence_mwh_lower_bound": 2.0,
                    "lost_energy_mwh": 1.0,
                    "family_day_dispatch_increment_mwh_lower_bound": 5.0,
                    "lost_energy_block_reason": "no_dispatch_truth",
                },
            ]
        )

        gap_family = build_fact_dispatch_source_gap_family_daily(truth, daily_gap)
        self.assertEqual(len(gap_family), 2)

        first = gap_family.iloc[0]
        second = gap_family.iloc[1]

        self.assertEqual(first["bmu_family_key"], "SHRSW")
        self.assertEqual(first["family_source_gap_next_action"], "mapping_and_source_audit")
        self.assertEqual(int(first["day_family_rank_by_source_gap"]), 1)
        self.assertAlmostEqual(float(first["source_gap_share_of_remaining_qa_shortfall"]), 0.25)

        self.assertEqual(second["bmu_family_key"], "KYPEW")
        self.assertEqual(second["family_source_gap_next_action"], "expand_dispatch_source_beyond_family_window")
        self.assertEqual(int(second["day_family_rank_by_source_gap"]), 2)


if __name__ == "__main__":
    unittest.main()
