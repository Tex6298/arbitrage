import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

from day_ahead_constraint_boundary import build_fact_day_ahead_constraint_boundary_half_hourly


class DayAheadConstraintBoundaryTests(unittest.TestCase):
    def test_build_fact_day_ahead_constraint_boundary_half_hourly_parses_public_csv(self) -> None:
        resource = {
            "id": "boundary-resource",
            "name": "Day Ahead Constraint Flows and Limits",
            "url": "https://example.com/boundary.csv",
        }
        raw = pd.DataFrame(
            [
                {
                    "Constraint Group": "EC5",
                    "Date (GMT/BST)": "2024-10-01T10:00:00",
                    "Limit (MW)": 4000,
                    "Flow (MW)": 3980,
                },
                {
                    "Constraint Group": "B6",
                    "Date (GMT/BST)": "2024-10-01T10:30:00",
                    "Limit (MW)": 2500,
                    "Flow (MW)": 1200,
                },
            ]
        )

        with patch("day_ahead_constraint_boundary._datapackage_show", return_value={"result": {"resources": [resource]}}):
            with patch("day_ahead_constraint_boundary._fetch_csv", return_value=raw):
                fact = build_fact_day_ahead_constraint_boundary_half_hourly(
                    start_date=dt.date(2024, 10, 1),
                    end_date=dt.date(2024, 10, 1),
                )

        self.assertEqual(len(fact), 2)
        tight = fact[fact["boundary_key"] == "EC5"].iloc[0]
        available = fact[fact["boundary_key"] == "B6"].iloc[0]
        self.assertEqual(tight["boundary_state"], "constraint_boundary_tight")
        self.assertAlmostEqual(float(tight["remaining_headroom_mw"]), 20.0)
        self.assertEqual(available["boundary_state"], "constraint_boundary_available")
        self.assertAlmostEqual(float(available["utilization_ratio"]), 0.48)

    def test_build_fact_day_ahead_constraint_boundary_half_hourly_matches_ckan_machine_name_and_path(self) -> None:
        resource = {
            "id": "boundary-resource",
            "name": "day_ahead_constraint_flows_and_limits",
            "title": "Day Ahead Constraint Flows and Limits",
            "path": "https://example.com/boundary.csv",
        }
        raw = pd.DataFrame(
            [
                {
                    "Constraint Group": "EC5",
                    "Date (GMT/BST)": "2024-10-01T10:00:00",
                    "Limit (MW)": 4000,
                    "Flow (MW)": 3980,
                }
            ]
        )

        with patch("day_ahead_constraint_boundary._datapackage_show", return_value={"result": {"resources": [resource]}}):
            with patch("day_ahead_constraint_boundary._fetch_csv", return_value=raw):
                fact = build_fact_day_ahead_constraint_boundary_half_hourly(
                    start_date=dt.date(2024, 10, 1),
                    end_date=dt.date(2024, 10, 1),
                )

        self.assertEqual(len(fact), 1)
        self.assertEqual(fact.iloc[0]["source_document_url"], "https://example.com/boundary.csv")

    def test_build_fact_day_ahead_constraint_boundary_half_hourly_handles_single_fallback_day(self) -> None:
        for day in (dt.date(2024, 10, 27), dt.date(2025, 10, 26)):
            with self.subTest(day=day):
                raw = pd.DataFrame(
                    {
                        "Constraint Group": ["ERROEX"] * 8,
                        "Time": [
                            f"{day.isoformat()} 00:00",
                            f"{day.isoformat()} 00:30",
                            f"{day.isoformat()} 01:00",
                            f"{day.isoformat()} 01:00",
                            f"{day.isoformat()} 01:30",
                            f"{day.isoformat()} 01:30",
                            f"{day.isoformat()} 02:00",
                            f"{day.isoformat()} 02:30",
                        ],
                        "Limit": [100.0] * 8,
                        "Flow": [50.0] * 8,
                    }
                )
                resource = {"id": "test", "name": "test", "url": "https://example.invalid/boundary.csv"}

                with (
                    patch("day_ahead_constraint_boundary._resource_metadata", return_value=resource),
                    patch("day_ahead_constraint_boundary._fetch_csv", return_value=raw),
                ):
                    fact = build_fact_day_ahead_constraint_boundary_half_hourly(day, day)

                self.assertEqual(len(fact), 8)
                self.assertEqual(
                    list(fact["interval_start_utc"]),
                    [
                        pd.Timestamp(day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=23),
                        pd.Timestamp(day - dt.timedelta(days=1), tz="UTC") + pd.Timedelta(hours=23, minutes=30),
                        pd.Timestamp(day, tz="UTC"),
                        pd.Timestamp(day, tz="UTC") + pd.Timedelta(minutes=30),
                        pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=1),
                        pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=1, minutes=30),
                        pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=2),
                        pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=2, minutes=30),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
