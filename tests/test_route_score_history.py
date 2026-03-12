import unittest

import pandas as pd

from route_score_history import ROUTE_SCORE_TABLE, build_fact_route_score_hourly


def _sample_prices() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2024-09-30T23:00:00Z"])
    return pd.DataFrame(
        {
            "GB": [50.0],
            "FR": [92.0],
            "NL": [78.0],
            "DE": [90.0],
            "PL": [110.0],
            "CZ": [85.0],
        },
        index=index,
    )


class RouteScoreHistoryTests(unittest.TestCase):
    def test_build_fact_route_score_hourly_uses_reviewed_capacity_tier(self) -> None:
        prices = _sample_prices()
        gb_transfer_gate = pd.DataFrame(
            [
                {
                    "date": "2024-10-01",
                    "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "cluster_key": "east_anglia_offshore",
                    "cluster_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "hub_key": "britned",
                    "hub_label": "BritNed",
                    "hub_target_zone": "NL",
                    "hub_neighbor_domain_key": "NL",
                    "hub_current_route_fit": "current",
                    "transfer_gate_mw_proxy": 500.0,
                    "transfer_gate_utilization_proxy": 0.49,
                    "gate_state": "capacity_unknown_reachable",
                    "gate_reason": "Transfer remains reachable, but first-pass border capacity is unpublished.",
                }
            ]
        )
        flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 100.0,
                }
            ]
        )
        review_policy = pd.DataFrame(
            [
                {
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_explicit_daily",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_explicit_daily",
                }
            ]
        )
        reviewed_capacity = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "interval_end_utc": "2024-10-01T00:00:00Z",
                    "direction_key": "gb_to_neighbor",
                    "border_key": "GB-NL",
                    "offered_capacity_mw": 700.0,
                }
            ]
        )

        fact = build_fact_route_score_hourly(
            prices=prices,
            gb_transfer_gate=gb_transfer_gate,
            interconnector_flow=flow,
            interconnector_capacity=None,
            interconnector_capacity_reviewed=reviewed_capacity,
            interconnector_capacity_review_policy=review_policy,
        )

        self.assertEqual(set(fact["route_name"]), {"R2_netback_GB_NL_DE_PL"})
        row = fact.iloc[0]
        self.assertEqual(row["route_delivery_tier"], "reviewed")
        self.assertEqual(row["route_delivery_signal"], "EXPORT_REVIEWED")
        self.assertEqual(row["capacity_policy_action"], "allow_reviewed_explicit_daily")
        self.assertAlmostEqual(float(row["deliverable_mw_proxy"]), 500.0)
        self.assertEqual(row["reviewed_border_gate_state"], "pass")

    def test_build_fact_route_score_hourly_blocks_internal_transfer_before_capacity(self) -> None:
        prices = _sample_prices()
        gb_transfer_gate = pd.DataFrame(
            [
                {
                    "date": "2024-10-01",
                    "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "cluster_key": "shetland_wind",
                    "cluster_label": "Shetland Wind",
                    "parent_region": "Scotland",
                    "hub_key": "britned",
                    "hub_label": "BritNed",
                    "hub_target_zone": "NL",
                    "hub_neighbor_domain_key": "NL",
                    "hub_current_route_fit": "current",
                    "transfer_gate_mw_proxy": 0.0,
                    "transfer_gate_utilization_proxy": 0.0,
                    "gate_state": "blocked_upstream_dependency",
                    "gate_reason": "Upstream dependency blocks the route.",
                }
            ]
        )

        fact = build_fact_route_score_hourly(
            prices=prices,
            gb_transfer_gate=gb_transfer_gate,
            interconnector_flow=None,
            interconnector_capacity=None,
            interconnector_capacity_reviewed=None,
            interconnector_capacity_review_policy=None,
        )

        self.assertEqual(len(fact), 1)
        row = fact.iloc[0]
        self.assertEqual(row["route_delivery_tier"], "blocked_internal_transfer")
        self.assertEqual(row["route_delivery_signal"], "HOLD")
        self.assertTrue(pd.isna(row["deliverable_route_score_eur_per_mwh"]))
        self.assertEqual(row["transfer_gate_state"], "blocked_upstream_dependency")

    def test_build_fact_route_score_hourly_caps_france_unknown_delivery_with_connector_proxy(self) -> None:
        prices = pd.DataFrame(
            {
                "GB": [50.0],
                "FR": [120.0],
                "NL": [78.0],
                "DE": [130.0],
                "PL": [160.0],
                "CZ": [85.0],
            },
            index=pd.DatetimeIndex(["2024-09-30T23:00:00Z"]),
        )
        gb_transfer_gate = pd.DataFrame(
            [
                {
                    "date": "2024-10-01",
                    "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "cluster_key": "east_anglia_offshore",
                    "cluster_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "hub_key": "ifa2",
                    "hub_label": "IFA2",
                    "hub_target_zone": "FR",
                    "hub_neighbor_domain_key": "FR",
                    "hub_current_route_fit": "current",
                    "transfer_gate_mw_proxy": 900.0,
                    "transfer_gate_utilization_proxy": 0.88,
                    "gate_state": "capacity_unknown_conditional",
                    "gate_reason": "Transfer remains plausible, but GB-FR capacity is unpublished.",
                }
            ]
        )
        france_connector = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "connector_key": "ifa2",
                    "connector_label": "IFA2",
                    "operator_name": "National Grid IFA2 Limited / RTE",
                    "nominal_capacity_mw": 1000.0,
                    "nominal_capacity_share_of_border": 0.25,
                    "connector_capacity_evidence_tier": "nominal_static",
                    "connector_headroom_proxy_mw": 350.0,
                    "connector_gate_state": "nominal_headroom_proxy",
                    "connector_gate_reason": "Nominal-share proxy only.",
                }
            ]
        )

        fact = build_fact_route_score_hourly(
            prices=prices,
            gb_transfer_gate=gb_transfer_gate,
            interconnector_flow=None,
            interconnector_capacity=None,
            interconnector_capacity_reviewed=None,
            interconnector_capacity_review_policy=None,
            france_connector=france_connector,
        )

        self.assertEqual(len(fact), 1)
        row = fact.iloc[0]
        self.assertEqual(row["route_name"], "R1_netback_GB_FR_DE_PL")
        self.assertEqual(row["route_delivery_tier"], "capacity_unknown")
        self.assertEqual(row["route_delivery_signal"], "EXPORT_CAPACITY_UNKNOWN")
        self.assertAlmostEqual(float(row["deliverable_mw_proxy"]), 350.0)
        self.assertEqual(row["connector_key"], "ifa2")
        self.assertEqual(row["connector_gate_state"], "nominal_headroom_proxy")

    def test_build_fact_route_score_hourly_blocks_when_operator_outage_zeros_france_connector(self) -> None:
        prices = pd.DataFrame(
            {
                "GB": [50.0],
                "FR": [120.0],
                "NL": [78.0],
                "DE": [130.0],
                "PL": [160.0],
                "CZ": [85.0],
            },
            index=pd.DatetimeIndex(["2024-09-30T23:00:00Z"]),
        )
        gb_transfer_gate = pd.DataFrame(
            [
                {
                    "date": "2024-10-01",
                    "interval_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "interval_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "interval_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "interval_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "cluster_key": "east_anglia_offshore",
                    "cluster_label": "East Anglia Offshore",
                    "parent_region": "England/Wales",
                    "hub_key": "ifa2",
                    "hub_label": "IFA2",
                    "hub_target_zone": "FR",
                    "hub_neighbor_domain_key": "FR",
                    "hub_current_route_fit": "current",
                    "transfer_gate_mw_proxy": 900.0,
                    "transfer_gate_utilization_proxy": 0.88,
                    "gate_state": "capacity_unknown_conditional",
                    "gate_reason": "Transfer remains plausible, but GB-FR capacity is unpublished.",
                }
            ]
        )
        france_connector = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "connector_key": "ifa2",
                    "connector_label": "IFA2",
                    "operator_name": "National Grid IFA2 Limited / RTE",
                    "operator_source_provider": "elexon_remit",
                    "operator_availability_state": "outage",
                    "operator_capacity_evidence_tier": "operator_outage_truth",
                    "operator_capacity_limit_mw": 0.0,
                    "nominal_capacity_mw": 1000.0,
                    "nominal_capacity_share_of_border": 0.25,
                    "connector_capacity_evidence_tier": "operator_outage_truth",
                    "connector_headroom_proxy_mw": 0.0,
                    "connector_gate_state": "operator_outage_blocked",
                    "connector_gate_reason": "Operator outage messages report the connector unavailable for the hour.",
                }
            ]
        )

        fact = build_fact_route_score_hourly(
            prices=prices,
            gb_transfer_gate=gb_transfer_gate,
            interconnector_flow=None,
            interconnector_capacity=None,
            interconnector_capacity_reviewed=None,
            interconnector_capacity_review_policy=None,
            france_connector=france_connector,
        )

        row = fact.iloc[0]
        self.assertEqual(row["route_delivery_tier"], "blocked_connector_capacity")
        self.assertEqual(row["route_delivery_signal"], "HOLD")
        self.assertTrue(pd.isna(row["deliverable_route_score_eur_per_mwh"]))
        self.assertEqual(row["connector_operator_availability_state"], "outage")


if __name__ == "__main__":
    unittest.main()
