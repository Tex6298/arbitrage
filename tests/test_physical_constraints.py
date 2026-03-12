import unittest

import pandas as pd

from physical_constraints import compute_netbacks


def _sample_prices() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2024-10-01T00:00:00Z"])
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


class PhysicalConstraintsTests(unittest.TestCase):
    def test_compute_netbacks_marks_confirmed_export_when_capacity_published(self) -> None:
        prices = _sample_prices()
        flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 400.0,
                },
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 300.0,
                },
            ]
        )
        capacity = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "offered_capacity_mw": 1000.0,
                },
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "offered_capacity_mw": 900.0,
                },
            ]
        )

        out = compute_netbacks(prices, interconnector_flow=flow, interconnector_capacity=capacity)
        row = out.iloc[0]
        self.assertEqual(row["R1_netback_GB_FR_DE_PL_gb_border_network_gate_state"], "pass")
        self.assertEqual(row["export_signal_network"], "EXPORT_CONFIRMED")
        self.assertEqual(row["best_route_network_confirmed"], "GB->NL->DE->PL")

    def test_compute_netbacks_marks_capacity_unknown_without_blocking_relaxed_route(self) -> None:
        prices = _sample_prices()
        flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 0.0,
                }
            ]
        )

        out = compute_netbacks(prices, interconnector_flow=flow, interconnector_capacity=None)
        row = out.iloc[0]
        self.assertEqual(row["R1_netback_GB_FR_DE_PL_gb_border_network_gate_state"], "capacity_unknown")
        self.assertEqual(row["export_signal_network"], "EXPORT_CAPACITY_UNKNOWN")
        self.assertTrue(pd.isna(row["best_netback_network_confirmed"]))
        self.assertEqual(row["best_route_network_relaxed"], "GB->NL->DE->PL")
        self.assertEqual(row["best_route_network_gate_state"], "capacity_unknown")

    def test_compute_netbacks_prefers_confirmed_route_when_best_price_route_is_blocked(self) -> None:
        prices = _sample_prices()
        flow = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 1000.0,
                },
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "signed_flow_from_gb_mw": 100.0,
                },
            ]
        )
        capacity = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-FR",
                    "direction_key": "gb_to_neighbor",
                    "offered_capacity_mw": 800.0,
                },
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "border_key": "GB-NL",
                    "direction_key": "gb_to_neighbor",
                    "offered_capacity_mw": 900.0,
                },
            ]
        )

        out = compute_netbacks(prices, interconnector_flow=flow, interconnector_capacity=capacity)
        row = out.iloc[0]
        self.assertEqual(row["R1_netback_GB_FR_DE_PL_gb_border_network_gate_state"], "blocked_headroom_proxy")
        self.assertEqual(row["R2_netback_GB_NL_DE_PL_gb_border_network_gate_state"], "pass")
        self.assertEqual(row["best_route_network_confirmed"], "GB->NL->DE->PL")
        self.assertEqual(row["export_signal_network"], "EXPORT_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
