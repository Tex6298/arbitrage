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


if __name__ == "__main__":
    unittest.main()
