import unittest

import pandas as pd

from curtailment_opportunity import (
    CURTAILMENT_OPPORTUNITY_TABLE,
    build_fact_curtailment_opportunity_hourly,
    materialize_curtailment_opportunity_history,
)


def _route_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-10-01").date(),
                "interval_start_local": pd.Timestamp("2024-10-01T10:00:00+01:00"),
                "interval_end_local": pd.Timestamp("2024-10-01T11:00:00+01:00"),
                "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                "interval_end_utc": pd.Timestamp("2024-10-01T10:00:00Z"),
                "cluster_key": "east_anglia_offshore",
                "cluster_label": "East Anglia Offshore",
                "parent_region": "England/Wales",
                "hub_key": "eleclink",
                "hub_label": "ElecLink",
                "route_name": "R1_netback_GB_FR_DE_PL",
                "route_label": "GB->FR->DE->PL",
                "route_border_key": "GB-FR",
                "route_target_zone": "FR",
                "route_price_score_eur_per_mwh": 47.5,
                "route_price_feasible_flag": True,
                "route_price_bottleneck": "GB->FR",
                "route_delivery_tier": "reviewed",
                "route_delivery_signal": "EXPORT_REVIEWED",
                "route_delivery_reason": "Reviewed public cap exists.",
                "deliverable_mw_proxy": 80.0,
                "deliverable_route_score_eur_per_mwh": 45.0,
                "internal_transfer_evidence_tier": "reviewed_internal_transfer_period",
                "internal_transfer_gate_state": "reviewed_pass_restricted",
                "internal_transfer_capacity_limit_mw": 85.0,
                "internal_transfer_source_provider": "public_reviewed_doc",
                "internal_transfer_source_key": "internal_boundary_restriction",
                "connector_notice_state": "upcoming",
                "connector_notice_known_flag": True,
                "connector_notice_active_flag": False,
                "connector_notice_upcoming_flag": True,
                "connector_notice_hours_until_start": 3.0,
                "connector_notice_hours_since_publication": 12.0,
                "connector_notice_lead_time_hours": 15.0,
                "connector_notice_revision_count": 2,
                "connector_notice_source_key": "eleclink_ntc_restriction",
                "connector_itl_state": "published_restriction",
                "connector_itl_capacity_limit_mw": 80.0,
                "connector_itl_source_key": "neso_interconnector_itl",
                "connector_gate_state": "reviewed_publication_cap",
                "connector_capacity_evidence_tier": "reviewed_public_doc_period",
                "reviewed_publication_state": "partial_capacity",
            }
        ]
    )


class CurtailmentOpportunityTests(unittest.TestCase):
    def test_build_fact_curtailment_opportunity_hourly_uses_proxy_and_notice_context(self) -> None:
        route_score = _route_rows()
        regional_proxy = pd.DataFrame(
            [
                {
                    "scope_type": "cluster",
                    "scope_key": "east_anglia_offshore",
                    "scope_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "hourly_curtailment_proxy_mwh": 120.0,
                    "hourly_curtailment_proxy_cost_gbp": 5000.0,
                }
            ]
        )
        upstream_market_state = pd.DataFrame(
            [
                {
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T10:00:00Z"),
                    "route_name": "R1_netback_GB_FR_DE_PL",
                    "source_provider": "manual_market_state",
                    "source_family": "manual_route_curve",
                    "source_key": "manual_fr_curve",
                    "source_published_utc": pd.Timestamp("2024-10-01T06:00:00Z"),
                    "forward_price_eur_per_mwh": 42.0,
                    "day_ahead_price_eur_per_mwh": 44.0,
                    "intraday_price_eur_per_mwh": 49.0,
                    "imbalance_price_eur_per_mwh": pd.NA,
                    "forward_to_day_ahead_spread_eur_per_mwh": 2.0,
                    "day_ahead_to_intraday_spread_eur_per_mwh": 5.0,
                    "forward_to_day_ahead_spread_bucket": "spread_flat",
                    "day_ahead_to_intraday_spread_bucket": "spread_positive",
                    "upstream_market_state": "intraday_stronger_than_day_ahead",
                    "upstream_market_state_feed_available_flag": True,
                }
            ]
        )
        system_balance_market_state = pd.DataFrame(
            [
                {
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T10:00:00Z"),
                    "date": pd.Timestamp("2024-10-01").date(),
                    "interval_start_local": pd.Timestamp("2024-10-01T10:00:00+01:00"),
                    "interval_end_local": pd.Timestamp("2024-10-01T11:00:00+01:00"),
                    "system_balance_source_provider": "elexon",
                    "system_balance_source_family": "public_system_balance",
                    "system_balance_source_key": "IMBALNGC|INDDEM|INDGEN|MELNGC",
                    "system_balance_source_dataset_keys": "IMBALNGC|INDDEM|INDGEN|MELNGC",
                    "system_balance_source_published_utc": pd.Timestamp("2024-10-01T08:30:00Z"),
                    "system_balance_feed_available_flag": True,
                    "system_balance_known_flag": True,
                    "system_balance_active_flag": True,
                    "system_balance_state": "tight_margin_and_active_imbalance",
                    "system_balance_imbalance_mw": 1200.0,
                    "system_balance_indicated_demand_mw": 42000.0,
                    "system_balance_indicated_generation_mw": 40100.0,
                    "system_balance_indicated_margin_mw": 900.0,
                    "system_balance_demand_minus_generation_mw": 1900.0,
                    "system_balance_margin_ratio": 900.0 / 42000.0,
                    "system_balance_imbalance_direction_bucket": "imbalance_strong_positive",
                    "system_balance_margin_direction_bucket": "margin_very_tight",
                    "source_lineage": "elexon:IMBALNGC|INDDEM|INDGEN|MELNGC",
                }
            ]
        )

        fact = build_fact_curtailment_opportunity_hourly(
            fact_route_score_hourly=route_score,
            fact_regional_curtailment_hourly_proxy=regional_proxy,
            fact_upstream_market_state_hourly=upstream_market_state,
            fact_system_balance_market_state_hourly=system_balance_market_state,
            truth_profile="proxy",
        )

        row = fact.iloc[0]
        self.assertEqual(row["curtailment_source_tier"], "regional_proxy")
        self.assertAlmostEqual(float(row["curtailment_selected_mwh"]), 120.0)
        self.assertAlmostEqual(float(row["opportunity_deliverable_mwh"]), 80.0)
        self.assertAlmostEqual(float(row["opportunity_spill_mwh"]), 40.0)
        self.assertAlmostEqual(float(row["opportunity_gross_value_eur"]), 3600.0)
        self.assertEqual(row["opportunity_state"], "curtailment_export_reviewed")
        self.assertAlmostEqual(float(row["route_price_score_eur_per_mwh"]), 47.5)
        self.assertTrue(bool(row["route_price_feasible_flag"]))
        self.assertEqual(row["route_price_bottleneck"], "GB->FR")
        self.assertTrue(bool(row["upstream_market_state_feed_available_flag"]))
        self.assertEqual(row["upstream_market_state"], "intraday_stronger_than_day_ahead")
        self.assertAlmostEqual(float(row["upstream_day_ahead_price_eur_per_mwh"]), 44.0)
        self.assertAlmostEqual(float(row["upstream_intraday_price_eur_per_mwh"]), 49.0)
        self.assertEqual(row["upstream_day_ahead_to_intraday_spread_bucket"], "spread_positive")
        self.assertEqual(row["upstream_market_state_source_key"], "manual_fr_curve")
        self.assertTrue(bool(row["system_balance_feed_available_flag"]))
        self.assertTrue(bool(row["system_balance_known_flag"]))
        self.assertEqual(row["system_balance_state"], "tight_margin_and_active_imbalance")
        self.assertEqual(row["system_balance_imbalance_direction_bucket"], "imbalance_strong_positive")
        self.assertEqual(row["system_balance_margin_direction_bucket"], "margin_very_tight")
        self.assertEqual(row["system_balance_source_provider"], "elexon")
        self.assertEqual(row["cluster_mapping_confidence"], "medium")
        self.assertIn("britned", str(row["cluster_preferred_hub_candidates"]).lower())
        self.assertTrue(str(row["cluster_curation_version"]).startswith("phase2_spatial_truth_v1"))
        self.assertFalse(bool(row["connector_capacity_tight_now_flag"]))
        self.assertTrue(bool(row["market_knew_connector_restriction_flag"]))
        self.assertEqual(row["connector_notice_market_state"], "known_upcoming_restriction")
        self.assertEqual(row["internal_transfer_evidence_tier"], "reviewed_internal_transfer_period")
        self.assertEqual(row["internal_transfer_gate_state"], "reviewed_pass_restricted")
        self.assertAlmostEqual(float(row["internal_transfer_capacity_limit_mw"]), 85.0)
        self.assertEqual(row["internal_transfer_source_provider"], "public_reviewed_doc")
        self.assertEqual(row["internal_transfer_source_key"], "internal_boundary_restriction")
        self.assertEqual(row["connector_itl_state"], "published_restriction")
        self.assertAlmostEqual(float(row["connector_itl_capacity_limit_mw"]), 80.0)
        self.assertEqual(row["connector_itl_source_key"], "neso_interconnector_itl")

    def test_build_fact_curtailment_opportunity_hourly_truth_overrides_proxy_when_available(self) -> None:
        route_score = _route_rows()
        regional_proxy = pd.DataFrame(
            [
                {
                    "scope_type": "cluster",
                    "scope_key": "east_anglia_offshore",
                    "scope_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "hourly_curtailment_proxy_mwh": 120.0,
                    "hourly_curtailment_proxy_cost_gbp": 5000.0,
                }
            ]
        )
        truth = pd.DataFrame(
            [
                {
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "elexon_bm_unit": "T_TEST-1",
                    "cluster_key": "east_anglia_offshore",
                    "lost_energy_mwh": 30.0,
                    "dispatch_down_evidence_mwh_lower_bound": 45.0,
                    "research_profile_include": True,
                    "precision_profile_include": False,
                    "lost_energy_estimate_flag": True,
                },
                {
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:30:00Z"),
                    "elexon_bm_unit": "T_TEST-2",
                    "cluster_key": "east_anglia_offshore",
                    "lost_energy_mwh": 20.0,
                    "dispatch_down_evidence_mwh_lower_bound": 35.0,
                    "research_profile_include": True,
                    "precision_profile_include": False,
                    "lost_energy_estimate_flag": True,
                },
            ]
        )

        fact = build_fact_curtailment_opportunity_hourly(
            fact_route_score_hourly=route_score,
            fact_regional_curtailment_hourly_proxy=regional_proxy,
            fact_bmu_curtailment_truth_half_hourly=truth,
            truth_profile="research",
        )

        row = fact.iloc[0]
        self.assertEqual(row["curtailment_source_tier"], "cluster_truth_research")
        self.assertAlmostEqual(float(row["curtailment_truth_mwh"]), 50.0)
        self.assertAlmostEqual(float(row["curtailment_truth_dispatch_mwh_lower_bound"]), 80.0)
        self.assertEqual(int(row["curtailment_truth_half_hour_count"]), 2)
        self.assertAlmostEqual(float(row["curtailment_selected_mwh"]), 50.0)
        self.assertAlmostEqual(float(row["opportunity_deliverable_mwh"]), 50.0)
        self.assertFalse(bool(row["curtailment_source_target_is_proxy"]))

    def test_build_fact_curtailment_opportunity_hourly_flags_active_known_tight_capacity(self) -> None:
        route_score = _route_rows()
        route_score.loc[0, "route_delivery_tier"] = "blocked_connector_capacity"
        route_score.loc[0, "route_delivery_signal"] = "HOLD"
        route_score.loc[0, "deliverable_mw_proxy"] = 0.0
        route_score.loc[0, "deliverable_route_score_eur_per_mwh"] = pd.NA
        route_score.loc[0, "connector_notice_state"] = "active"
        route_score.loc[0, "connector_notice_active_flag"] = True
        route_score.loc[0, "connector_notice_upcoming_flag"] = False

        regional_proxy = pd.DataFrame(
            [
                {
                    "scope_type": "cluster",
                    "scope_key": "east_anglia_offshore",
                    "scope_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "hourly_curtailment_proxy_mwh": 60.0,
                    "hourly_curtailment_proxy_cost_gbp": 2500.0,
                }
            ]
        )

        fact = build_fact_curtailment_opportunity_hourly(
            fact_route_score_hourly=route_score,
            fact_regional_curtailment_hourly_proxy=regional_proxy,
            truth_profile="proxy",
        )

        row = fact.iloc[0]
        self.assertEqual(row["opportunity_state"], "curtailment_blocked_connector_capacity")
        self.assertTrue(bool(row["connector_capacity_tight_now_flag"]))
        self.assertTrue(bool(row["market_knew_connector_restriction_flag"]))
        self.assertEqual(row["connector_notice_market_state"], "tight_now_and_publicly_known")
        self.assertAlmostEqual(float(row["opportunity_deliverable_mwh"]), 0.0)

    def test_materialize_curtailment_opportunity_history_writes_csv(self) -> None:
        route_score = _route_rows()
        regional_proxy = pd.DataFrame(
            [
                {
                    "scope_type": "cluster",
                    "scope_key": "east_anglia_offshore",
                    "scope_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "interval_start_utc": pd.Timestamp("2024-10-01T09:00:00Z"),
                    "hourly_curtailment_proxy_mwh": 120.0,
                    "hourly_curtailment_proxy_cost_gbp": 5000.0,
                }
            ]
        )

        with self.subTest("materialize"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp_dir:
                frames = materialize_curtailment_opportunity_history(
                    output_dir=tmp_dir,
                    fact_route_score_hourly=route_score,
                    fact_regional_curtailment_hourly_proxy=regional_proxy,
                    truth_profile="proxy",
                )
                self.assertEqual(set(frames), {CURTAILMENT_OPPORTUNITY_TABLE})
                self.assertTrue((Path(tmp_dir) / f"{CURTAILMENT_OPPORTUNITY_TABLE}.csv").exists())


if __name__ == "__main__":
    unittest.main()
