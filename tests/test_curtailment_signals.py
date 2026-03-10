import unittest

import pandas as pd

from curtailment_signals import CONSTRAINT_QA_TARGET_DEFINITION, add_constraint_qa_columns


class CurtailmentSignalsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
