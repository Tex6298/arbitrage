import unittest

import pandas as pd

from truth_store_focus import (
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


if __name__ == "__main__":
    unittest.main()
