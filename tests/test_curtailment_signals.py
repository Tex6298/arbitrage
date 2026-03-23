import datetime as dt
import http.client
import unittest
from unittest.mock import patch

import pandas as pd

from bmu_truth_utils import build_half_hour_interval_frame
from curtailment_signals import (
    CONSTRAINT_QA_TARGET_DEFINITION,
    _fetch_bytes,
    _fetch_csv,
    add_constraint_qa_columns,
    build_regional_curtailment_hourly_proxy,
    fetch_constraint_daily,
)


class CurtailmentSignalsTests(unittest.TestCase):
    def test_fetch_bytes_retries_remote_disconnect(self) -> None:
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"ok"

        with patch(
            "curtailment_signals.urllib.request.urlopen",
            side_effect=[http.client.RemoteDisconnected("closed"), _Response()],
        ):
            payload = _fetch_bytes("https://example.com/test.csv")

        self.assertEqual(payload, b"ok")

    def test_fetch_csv_falls_back_from_utf8_to_cp1252(self) -> None:
        payload = "col\nalpha\xa0beta\n".encode("cp1252")

        with patch("curtailment_signals._fetch_bytes", return_value=payload):
            frame = _fetch_csv("https://example.com/test.csv")

        self.assertEqual(frame.iloc[0]["col"], "alpha\xa0beta")

    def test_constraint_qa_columns_clip_negative_thermal_without_touching_raw_total(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-10-03").date(),
                    "total_curtailment_mwh": -989.0,
                    "voltage_constraints_volume_mwh": 7556.0,
                    "thermal_constraints_volume_mwh": -8545.0,
                    "increasing_system_inertia_volume_mwh": 0.0,
                    "reducing_largest_loss_volume_mwh": 0.0,
                }
            ]
        )

        enriched = add_constraint_qa_columns(raw)
        first_row = enriched.iloc[0]

        self.assertEqual(first_row["qa_target_definition"], CONSTRAINT_QA_TARGET_DEFINITION)
        self.assertEqual(first_row["total_curtailment_mwh"], -989.0)
        self.assertEqual(first_row["qa_wind_voltage_positive_mwh"], 7556.0)
        self.assertEqual(first_row["qa_wind_thermal_positive_mwh"], 0.0)
        self.assertEqual(first_row["qa_wind_relevant_positive_mwh"], 7556.0)
        self.assertEqual(first_row["qa_inertia_positive_mwh"], 0.0)
        self.assertEqual(first_row["qa_largest_loss_positive_mwh"], 0.0)

    def test_fetch_constraint_daily_supports_2023_2024_resource(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "Date": "2024-03-15",
                    "Largest Loss Volume": 1.0,
                    "Largest Loss Cost": 10.0,
                    "System Inertia Volume": 2.0,
                    "System Inertia Cost": 20.0,
                    "Voltage Volume": 3.0,
                    "Voltage Cost": 30.0,
                    "Thermal Volume": 4.0,
                    "Thermal Cost": 40.0,
                }
            ]
        )

        with patch("curtailment_signals._fetch_csv", return_value=raw) as fetch_csv:
            fact = fetch_constraint_daily("2023-2024")

        self.assertIn("constraint-breakdown-2023-2024.csv", fetch_csv.call_args.args[0])
        self.assertEqual(fact.iloc[0]["source_year_label"], "2023-2024")
        self.assertEqual(fact.iloc[0]["source_resource_id"], "24d067d8-1328-452a-9720-21cb691e491e")
        self.assertEqual(float(fact.iloc[0]["total_curtailment_mwh"]), 10.0)
        self.assertEqual(float(fact.iloc[0]["total_curtailment_cost_gbp"]), 100.0)

    def test_build_regional_curtailment_hourly_proxy_handles_fallback_dst_days(self) -> None:
        for day in (dt.date(2024, 10, 27), dt.date(2025, 10, 26)):
            with self.subTest(day=day):
                intervals = build_half_hour_interval_frame(day, day).rename(columns={"settlement_date": "date"})
                wind_split = intervals[["date", "interval_start_local", "interval_start_utc"]].copy()
                wind_split["scotland_wind_mw"] = 30.0
                wind_split["england_wales_wind_mw"] = 70.0
                constraints = pd.DataFrame(
                    [
                        {
                            "date": day,
                            "source_year_label": f"{day.year}-{day.year + 1}",
                            "total_curtailment_mwh": 25.0,
                            "total_curtailment_cost_gbp": 2500.0,
                        }
                    ]
                )

                fact = build_regional_curtailment_hourly_proxy(constraints, wind_split)
                parent_region = fact[fact["scope_type"] == "parent_region"].copy()

                self.assertEqual(parent_region["interval_start_utc"].nunique(), 25)
                self.assertEqual(len(parent_region), 50)

                repeated_hour_utc = list(
                    parent_region[parent_region["interval_start_local"].dt.hour == 1]["interval_start_utc"]
                    .drop_duplicates()
                    .sort_values()
                )
                self.assertEqual(
                    repeated_hour_utc,
                    [
                        pd.Timestamp(f"{day.isoformat()}T00:00:00Z"),
                        pd.Timestamp(f"{day.isoformat()}T01:00:00Z"),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
