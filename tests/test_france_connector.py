import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from france_connector import (
    DIM_INTERCONNECTOR_CABLE_TABLE,
    FRANCE_CONNECTOR_TABLE,
    build_fact_france_connector_hourly,
    interconnector_cable_frame,
    materialize_france_connector_history,
)


class FranceConnectorTests(unittest.TestCase):
    def test_interconnector_cable_frame_exposes_current_france_connectors(self) -> None:
        dim = interconnector_cable_frame()

        self.assertEqual(set(dim["connector_key"]), {"ifa", "ifa2", "eleclink"})
        eleclink = dim[dim["connector_key"] == "eleclink"].iloc[0]
        self.assertEqual(eleclink["current_route_fit"], "current")
        self.assertAlmostEqual(float(dim["nominal_capacity_mw"].sum()), 4000.0)

    def test_build_fact_france_connector_hourly_splits_border_flow_by_nominal_share(self) -> None:
        interconnector_flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 1200.0,
                }
            ]
        )

        fact = build_fact_france_connector_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            interconnector_flow=interconnector_flow,
            interconnector_capacity=None,
            interconnector_capacity_review_policy=None,
            interconnector_capacity_reviewed=None,
        )

        hour = fact[fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z")]
        self.assertEqual(len(hour), 3)
        ifa = hour[hour["connector_key"] == "ifa"].iloc[0]
        ifa2 = hour[hour["connector_key"] == "ifa2"].iloc[0]
        eleclink = hour[hour["connector_key"] == "eleclink"].iloc[0]

        self.assertAlmostEqual(float(ifa["connector_signed_flow_from_gb_mw_proxy"]), 600.0)
        self.assertAlmostEqual(float(ifa["connector_headroom_proxy_mw"]), 1400.0)
        self.assertEqual(ifa["connector_gate_state"], "nominal_headroom_proxy")

        self.assertAlmostEqual(float(ifa2["connector_signed_flow_from_gb_mw_proxy"]), 300.0)
        self.assertAlmostEqual(float(ifa2["connector_headroom_proxy_mw"]), 700.0)
        self.assertAlmostEqual(float(eleclink["connector_headroom_proxy_mw"]), 700.0)
        self.assertEqual(ifa2["capacity_policy_action"], "keep_capacity_unknown")

    def test_build_fact_france_connector_hourly_applies_operator_partial_capacity_cap(self) -> None:
        interconnector_flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 1200.0,
                }
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "connector_key": "ifa2",
                    "source_provider": "elexon_remit",
                    "operator_availability_state": "partial_outage",
                    "operator_capacity_evidence_tier": "operator_outage_truth",
                    "operator_capacity_limit_mw": 500.0,
                    "operator_source_gap_reason": pd.NA,
                }
            ]
        )

        fact = build_fact_france_connector_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            interconnector_flow=interconnector_flow,
            interconnector_capacity=None,
            interconnector_capacity_review_policy=None,
            interconnector_capacity_reviewed=None,
            france_connector_availability=availability,
        )

        ifa2 = fact[
            (fact["connector_key"] == "ifa2")
            & (fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z"))
        ].iloc[0]
        self.assertEqual(ifa2["operator_availability_state"], "partial_outage")
        self.assertEqual(ifa2["connector_gate_state"], "operator_partial_capacity_cap")
        self.assertAlmostEqual(float(ifa2["connector_headroom_proxy_mw"]), 500.0)

    def test_build_fact_france_connector_hourly_applies_reviewed_publication_cap(self) -> None:
        interconnector_flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 800.0,
                }
            ]
        )
        reviewed_period = pd.DataFrame(
            [
                {
                    "connector_key": "eleclink",
                    "connector_label": "ElecLink",
                    "direction_key": "gb_to_neighbor",
                    "reviewed_scope": "france_connector_public_doc_period",
                    "review_state": "accepted_reviewed_tier",
                    "reviewed_evidence_tier": "reviewed_public_doc_period",
                    "reviewed_tier_accepted_flag": True,
                    "capacity_policy_action": "allow_reviewed_public_period",
                    "reviewed_publication_state": "partial_capacity",
                    "period_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "period_end_utc": pd.Timestamp("2024-10-01T03:00:00Z"),
                    "period_timezone": "UTC",
                    "connector_nominal_capacity_mw": 1000.0,
                    "reviewed_capacity_limit_mw": 250.0,
                    "reviewed_available_capacity_mw": 250.0,
                    "reviewed_unavailable_capacity_mw": 750.0,
                    "source_provider": "public_reviewed_doc",
                    "source_family": "eleclink_public_doc",
                    "source_key": "eleclink_ntc_restriction",
                    "source_label": "ElecLink NTC restriction statement",
                    "source_document_title": "ElecLink NTC restriction",
                    "source_document_url": "https://www.eleclink.co.uk/publications/ntc-restrictions",
                    "source_reference": "EL-NTC-1",
                    "source_published_date": dt.date(2024, 9, 30),
                    "review_note": "Public restriction statement.",
                    "target_is_proxy": False,
                }
            ]
        )

        fact = build_fact_france_connector_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            interconnector_flow=interconnector_flow,
            interconnector_capacity=None,
            interconnector_capacity_review_policy=None,
            interconnector_capacity_reviewed=None,
            france_connector_reviewed_period=reviewed_period,
            france_connector_availability=None,
        )

        eleclink = fact[
            (fact["connector_key"] == "eleclink")
            & (fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z"))
        ].iloc[0]
        self.assertEqual(eleclink["connector_capacity_evidence_tier"], "reviewed_public_doc_period")
        self.assertEqual(eleclink["connector_gate_state"], "reviewed_publication_cap")
        self.assertEqual(eleclink["reviewed_publication_source_key"], "eleclink_ntc_restriction")
        self.assertAlmostEqual(float(eleclink["connector_headroom_proxy_mw"]), 250.0)

    def test_materialize_france_connector_history_writes_dimension_and_fact(self) -> None:
        interconnector_flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 0.0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_france_connector_history(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                output_dir=tmp_dir,
                token=None,
                interconnector_flow=interconnector_flow,
                interconnector_capacity=None,
                interconnector_capacity_review_policy=None,
                interconnector_capacity_reviewed=None,
            )

            self.assertEqual(set(frames), {DIM_INTERCONNECTOR_CABLE_TABLE, FRANCE_CONNECTOR_TABLE})
            self.assertTrue((Path(tmp_dir) / f"{DIM_INTERCONNECTOR_CABLE_TABLE}.csv").exists())
            self.assertTrue((Path(tmp_dir) / f"{FRANCE_CONNECTOR_TABLE}.csv").exists())


if __name__ == "__main__":
    unittest.main()
