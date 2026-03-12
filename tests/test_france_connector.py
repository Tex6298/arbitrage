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
