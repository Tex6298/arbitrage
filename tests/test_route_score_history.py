import datetime as dt
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
    def test_build_fact_route_score_hourly_reviewed_internal_transfer_overrides_proxy_gate(self) -> None:
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
                    "transfer_gate_mw_proxy": 0.0,
                    "transfer_gate_utilization_proxy": 0.0,
                    "gate_state": "blocked_upstream_dependency",
                    "gate_reason": "Proxy blocks the route.",
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
        reviewed_internal = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "interval_end_utc": "2024-10-01T00:00:00Z",
                    "cluster_key": "east_anglia_offshore",
                    "hub_key": "britned",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_internal_transfer_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_internal_period",
                    "reviewed_gate_state": "reviewed_pass_restricted",
                    "reviewed_capacity_limit_mw": 120.0,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "public_boundary_doc",
                    "source_key": "internal_boundary_restriction",
                }
            ]
        )

        fact = build_fact_route_score_hourly(
            prices=prices,
            gb_transfer_gate=gb_transfer_gate,
            interconnector_flow=None,
            interconnector_capacity=None,
            interconnector_capacity_reviewed=reviewed_capacity,
            interconnector_capacity_review_policy=review_policy,
            gb_transfer_reviewed_hourly=reviewed_internal,
        )

        self.assertEqual(len(fact), 1)
        row = fact.iloc[0]
        self.assertEqual(row["route_delivery_tier"], "reviewed")
        self.assertEqual(row["internal_transfer_evidence_tier"], "reviewed_internal_transfer_period")
        self.assertEqual(row["internal_transfer_gate_state"], "reviewed_pass_restricted")
        self.assertAlmostEqual(float(row["internal_transfer_capacity_limit_mw"]), 120.0)
        self.assertEqual(row["internal_transfer_source_provider"], "public_reviewed_doc")
        self.assertEqual(row["internal_transfer_source_key"], "internal_boundary_restriction")
        self.assertAlmostEqual(float(row["deliverable_mw_proxy"]), 120.0)

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

    def test_build_fact_route_score_hourly_prefers_tighter_boundary_reviewed_internal_row(self) -> None:
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
                    "transfer_gate_utilization_proxy": 0.5,
                    "gate_state": "capacity_unknown_reachable",
                    "gate_reason": "Proxy leaves the route reachable.",
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
        reviewed_internal = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "interval_end_utc": "2024-10-01T00:00:00Z",
                    "cluster_key": "east_anglia_offshore",
                    "hub_key": "britned",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_internal_transfer_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_internal_period",
                    "reviewed_gate_state": "reviewed_pass_restricted",
                    "reviewed_capacity_limit_mw": 200.0,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "public_boundary_doc",
                    "source_key": "internal_boundary_restriction",
                },
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "interval_end_utc": "2024-10-01T00:00:00Z",
                    "cluster_key": "east_anglia_offshore",
                    "hub_key": "britned",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_internal_constraint_boundary",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_boundary_day_ahead_gate",
                    "reviewed_gate_state": "reviewed_boundary_tight",
                    "reviewed_capacity_limit_mw": 80.0,
                    "source_provider": "neso",
                    "source_family": "day_ahead_constraint_boundary",
                    "source_key": "fact_day_ahead_constraint_boundary_half_hourly:FLOWSTH",
                },
            ]
        )

        fact = build_fact_route_score_hourly(
            prices=prices,
            gb_transfer_gate=gb_transfer_gate,
            interconnector_flow=None,
            interconnector_capacity=None,
            interconnector_capacity_reviewed=reviewed_capacity,
            interconnector_capacity_review_policy=review_policy,
            gb_transfer_reviewed_hourly=reviewed_internal,
        )

        row = fact.iloc[0]
        self.assertEqual(row["internal_transfer_evidence_tier"], "reviewed_internal_constraint_boundary")
        self.assertEqual(row["internal_transfer_gate_state"], "reviewed_boundary_tight")
        self.assertEqual(row["internal_transfer_source_provider"], "neso")
        self.assertEqual(
            row["internal_transfer_source_key"],
            "fact_day_ahead_constraint_boundary_half_hourly:FLOWSTH",
        )
        self.assertAlmostEqual(float(row["internal_transfer_capacity_limit_mw"]), 80.0)
        self.assertAlmostEqual(float(row["deliverable_mw_proxy"]), 80.0)

    def test_build_fact_route_score_hourly_uses_connector_itl_reviewed_tier(self) -> None:
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
                    "transfer_gate_mw_proxy": 300.0,
                    "transfer_gate_utilization_proxy": 0.40,
                    "gate_state": "capacity_unknown_reachable",
                    "gate_reason": "Transfer remains reachable, but border capacity is unpublished.",
                }
            ]
        )
        interconnector_itl = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "connector_key": "britned",
                    "connector_label": "BritNed",
                    "direction_key": "gb_to_neighbor",
                    "itl_state": "published_restriction",
                    "itl_mw": 120.0,
                    "auction_type": "Intraday 1",
                    "restriction_reason": "System Security",
                    "source_provider": "neso",
                    "source_key": "neso_interconnector_itl",
                    "source_published_utc": "2024-09-30T22:00:00Z",
                }
            ]
        )

        fact = build_fact_route_score_hourly(
            prices=prices,
            gb_transfer_gate=gb_transfer_gate,
            interconnector_itl=interconnector_itl,
            interconnector_flow=None,
            interconnector_capacity=None,
            interconnector_capacity_reviewed=None,
            interconnector_capacity_review_policy=None,
        )

        row = fact.iloc[0]
        self.assertEqual(row["route_delivery_tier"], "reviewed")
        self.assertEqual(row["route_delivery_signal"], "EXPORT_REVIEWED")
        self.assertEqual(row["connector_key"], "britned")
        self.assertEqual(row["connector_itl_state"], "published_restriction")
        self.assertEqual(row["connector_capacity_evidence_tier"], "neso_interconnector_itl")
        self.assertAlmostEqual(float(row["connector_itl_capacity_limit_mw"]), 120.0)
        self.assertAlmostEqual(float(row["deliverable_mw_proxy"]), 120.0)

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
        self.assertEqual(row["internal_transfer_evidence_tier"], "gb_topology_transfer_gate_proxy")
        self.assertEqual(row["internal_transfer_gate_state"], "blocked_upstream_dependency")

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
        france_connector_notice = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "interval_end_utc": "2024-10-01T00:00:00Z",
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "direction_key": "gb_to_neighbor",
                    "notice_state": "upcoming",
                    "notice_known_flag": True,
                    "notice_active_flag": False,
                    "notice_upcoming_flag": True,
                    "notice_group_key": "eleclink|gb_to_neighbor|2024-10-01T02:00:00Z|2024-10-01T12:00:00Z",
                    "notice_planning_state": "operational_restriction",
                    "planned_outage_flag": False,
                    "expected_capacity_limit_mw": 250.0,
                    "hours_until_notice_start": 3.0,
                    "days_until_notice_start": 3.0 / 24.0,
                    "hours_since_notice_publication": 13.0,
                    "notice_lead_time_hours": 16.0,
                    "notice_revision_count": 1,
                    "source_revision_rank": 1,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "eleclink_public_doc",
                    "source_key": "eleclink_ntc_restriction",
                    "source_label": "ElecLink NTC restriction statement",
                    "source_document_title": "ElecLink restriction",
                    "source_document_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
                    "source_reference": "EL-NTC-1",
                    "source_published_utc": "2024-09-30T10:00:00Z",
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
            france_connector_notice=france_connector_notice,
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
        france_connector_notice = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "interval_end_utc": "2024-10-01T00:00:00Z",
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "direction_key": "gb_to_neighbor",
                    "notice_state": "upcoming",
                    "notice_known_flag": True,
                    "notice_active_flag": False,
                    "notice_upcoming_flag": True,
                    "notice_group_key": "eleclink|gb_to_neighbor|2024-10-01T02:00:00Z|2024-10-01T12:00:00Z",
                    "notice_planning_state": "operational_restriction",
                    "planned_outage_flag": False,
                    "expected_capacity_limit_mw": 250.0,
                    "hours_until_notice_start": 3.0,
                    "days_until_notice_start": 3.0 / 24.0,
                    "hours_since_notice_publication": 13.0,
                    "notice_lead_time_hours": 16.0,
                    "notice_revision_count": 1,
                    "source_revision_rank": 1,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "eleclink_public_doc",
                    "source_key": "eleclink_ntc_restriction",
                    "source_label": "ElecLink NTC restriction statement",
                    "source_document_title": "ElecLink restriction",
                    "source_document_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
                    "source_reference": "EL-NTC-1",
                    "source_published_utc": "2024-09-30T10:00:00Z",
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
            france_connector_notice=france_connector_notice,
        )

        row = fact.iloc[0]
        self.assertEqual(row["route_delivery_tier"], "blocked_connector_capacity")
        self.assertEqual(row["route_delivery_signal"], "HOLD")
        self.assertTrue(pd.isna(row["deliverable_route_score_eur_per_mwh"]))
        self.assertEqual(row["connector_operator_availability_state"], "outage")

    def test_build_fact_route_score_hourly_uses_reviewed_france_connector_publication_tier(self) -> None:
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
                    "hub_key": "eleclink",
                    "hub_label": "ElecLink",
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
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "operator_name": "ElecLink Limited / Getlink",
                    "nominal_capacity_mw": 1000.0,
                    "nominal_capacity_share_of_border": 0.25,
                    "connector_capacity_evidence_tier": "reviewed_public_doc_period",
                    "connector_headroom_proxy_mw": 250.0,
                    "connector_gate_state": "reviewed_publication_cap",
                    "connector_gate_reason": "Reviewed public doc cap.",
                    "reviewed_publication_state": "partial_capacity",
                    "reviewed_publication_evidence_tier": "reviewed_public_doc_period",
                    "reviewed_publication_tier_accepted_flag": True,
                    "reviewed_publication_capacity_policy_action": "allow_reviewed_public_period",
                    "reviewed_publication_capacity_limit_mw": 250.0,
                    "reviewed_publication_source_provider": "public_reviewed_doc",
                    "reviewed_publication_source_family": "eleclink_public_doc",
                    "reviewed_publication_source_key": "eleclink_ntc_restriction",
                    "reviewed_publication_source_label": "ElecLink NTC restriction statement",
                    "reviewed_publication_source_document_title": "ElecLink restriction",
                    "reviewed_publication_source_document_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
                    "reviewed_publication_source_reference": "EL-NTC-1",
                    "reviewed_publication_source_published_date": dt.date(2024, 9, 30),
                    "reviewed_publication_source_count": 1,
                }
            ]
        )
        france_connector_notice = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "interval_end_utc": "2024-10-01T00:00:00Z",
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "direction_key": "gb_to_neighbor",
                    "notice_state": "upcoming",
                    "notice_known_flag": True,
                    "notice_active_flag": False,
                    "notice_upcoming_flag": True,
                    "notice_group_key": "eleclink|gb_to_neighbor|2024-10-01T02:00:00Z|2024-10-01T12:00:00Z",
                    "notice_planning_state": "operational_restriction",
                    "planned_outage_flag": False,
                    "expected_capacity_limit_mw": 250.0,
                    "hours_until_notice_start": 3.0,
                    "days_until_notice_start": 3.0 / 24.0,
                    "hours_since_notice_publication": 13.0,
                    "notice_lead_time_hours": 16.0,
                    "notice_revision_count": 1,
                    "source_revision_rank": 1,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "eleclink_public_doc",
                    "source_key": "eleclink_ntc_restriction",
                    "source_label": "ElecLink NTC restriction statement",
                    "source_document_title": "ElecLink restriction",
                    "source_document_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
                    "source_reference": "EL-NTC-1",
                    "source_published_utc": "2024-09-30T10:00:00Z",
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
            france_connector_notice=france_connector_notice,
        )

        row = fact.iloc[0]
        self.assertEqual(row["route_delivery_tier"], "reviewed")
        self.assertEqual(row["route_delivery_signal"], "EXPORT_REVIEWED")
        self.assertEqual(row["reviewed_publication_source_key"], "eleclink_ntc_restriction")
        self.assertAlmostEqual(float(row["deliverable_mw_proxy"]), 250.0)
        self.assertEqual(row["connector_notice_state"], "upcoming")
        self.assertAlmostEqual(float(row["connector_notice_hours_until_start"]), 3.0)
        self.assertEqual(row["connector_notice_source_key"], "eleclink_ntc_restriction")


if __name__ == "__main__":
    unittest.main()
