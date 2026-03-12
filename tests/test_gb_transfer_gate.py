import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gb_transfer_gate import GB_TRANSFER_GATE_TABLE, build_fact_gb_transfer_gate_hourly, materialize_gb_transfer_gate_history


class GbTransferGateTests(unittest.TestCase):
    def test_build_fact_gb_transfer_gate_hourly_combines_reachability_and_border_overlay(self) -> None:
        flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 400.0,
                },
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-NO2",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 100.0,
                },
            ]
        )
        capacity = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "offered_capacity_mw": 1000.0,
                },
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-NO2",
                    "direction_key": "gb_to_neighbor",
                    "offered_capacity_mw": 900.0,
                },
            ]
        )

        fact = build_fact_gb_transfer_gate_hourly(
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            interconnector_flow=flow,
            interconnector_capacity=capacity,
        )

        self.assertFalse(fact.empty)

        nl_row = fact[
            (fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z"))
            & (fact["cluster_key"] == "east_anglia_offshore")
            & (fact["hub_key"] == "britned")
        ].iloc[0]
        self.assertEqual(nl_row["gate_state"], "pass")
        self.assertAlmostEqual(float(nl_row["structural_gate_mw_proxy"]), 918.0)
        self.assertAlmostEqual(float(nl_row["transfer_gate_mw_proxy"]), 600.0)
        self.assertEqual(nl_row["border_gate_state"], "pass")

        fr_row = fact[
            (fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z"))
            & (fact["cluster_key"] == "east_anglia_offshore")
            & (fact["hub_key"] == "ifa")
        ].iloc[0]
        self.assertEqual(fr_row["gate_state"], "capacity_unknown_conditional")
        self.assertEqual(fr_row["border_gate_state"], "capacity_unknown")

        shetland_row = fact[
            (fact["interval_start_utc"] == pd.Timestamp("2024-09-30T23:00:00Z"))
            & (fact["cluster_key"] == "shetland_wind")
            & (fact["hub_key"] == "nsl")
        ].iloc[0]
        self.assertEqual(shetland_row["gate_state"], "blocked_upstream_dependency")
        self.assertAlmostEqual(float(shetland_row["transfer_gate_mw_proxy"]), 0.0)

    def test_materialize_gb_transfer_gate_history_writes_csv(self) -> None:
        flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 400.0,
                }
            ]
        )
        capacity = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-09-30T23:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "offered_capacity_mw": 1000.0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = materialize_gb_transfer_gate_history(
                start_date=dt.date(2024, 10, 1),
                end_date=dt.date(2024, 10, 1),
                output_dir=tmp_dir,
                interconnector_flow=flow,
                interconnector_capacity=capacity,
            )

            self.assertIn(GB_TRANSFER_GATE_TABLE, frames)
            output_path = Path(tmp_dir) / f"{GB_TRANSFER_GATE_TABLE}.csv"
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
