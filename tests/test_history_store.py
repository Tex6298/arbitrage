import datetime as dt
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from history_store import ingest_truth_csv_tree_to_sqlite, upsert_truth_frames_to_sqlite


class HistoryStoreTests(unittest.TestCase):
    def test_upsert_truth_frames_to_sqlite_replaces_existing_primary_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "truth_store.sqlite"
            first = pd.DataFrame(
                [
                    {
                        "settlement_date": dt.date(2024, 10, 1),
                        "qa_reconciliation_status": "fail",
                        "gb_daily_estimated_lost_energy_mwh": 10.0,
                    }
                ]
            )
            second = pd.DataFrame(
                [
                    {
                        "settlement_date": dt.date(2024, 10, 1),
                        "qa_reconciliation_status": "warn",
                        "gb_daily_estimated_lost_energy_mwh": 12.5,
                    }
                ]
            )

            upsert_truth_frames_to_sqlite(
                {"fact_curtailment_reconciliation_daily": first},
                db_path=db_path,
            )
            summary = upsert_truth_frames_to_sqlite(
                {"fact_curtailment_reconciliation_daily": second},
                db_path=db_path,
            )

            self.assertEqual(int(summary.iloc[0]["rows_loaded"]), 1)
            self.assertEqual(int(summary.iloc[0]["table_row_count"]), 1)

            with closing(sqlite3.connect(db_path)) as connection:
                stored = pd.read_sql_query(
                    "SELECT settlement_date, qa_reconciliation_status, gb_daily_estimated_lost_energy_mwh "
                    "FROM fact_curtailment_reconciliation_daily",
                    connection,
                )
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored.iloc[0]["settlement_date"], "2024-10-01")
            self.assertEqual(stored.iloc[0]["qa_reconciliation_status"], "warn")
            self.assertAlmostEqual(float(stored.iloc[0]["gb_daily_estimated_lost_energy_mwh"]), 12.5)

    def test_ingest_truth_csv_tree_to_sqlite_dedupes_across_daily_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir) / "daily"
            day_one = root_dir / "2024-10-01"
            day_two = root_dir / "2024-10-02"
            day_one.mkdir(parents=True)
            day_two.mkdir(parents=True)
            db_path = Path(tmp_dir) / "truth_store.sqlite"

            pd.DataFrame(
                [
                    {
                        "settlement_date": "2024-10-01",
                        "bmu_family_key": "SGRWO",
                        "dispatch_minus_lost_energy_gap_mwh": 100.0,
                    }
                ]
            ).to_csv(day_one / "fact_bmu_family_shortfall_daily.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "settlement_date": "2024-10-01",
                        "bmu_family_key": "SGRWO",
                        "dispatch_minus_lost_energy_gap_mwh": 80.0,
                    }
                ]
            ).to_csv(day_two / "fact_bmu_family_shortfall_daily.csv", index=False)

            summary = ingest_truth_csv_tree_to_sqlite(root_dir=root_dir, db_path=db_path)
            self.assertEqual(int(summary.iloc[0]["files_loaded"]), 2)
            self.assertEqual(int(summary.iloc[0]["table_row_count"]), 1)

            with closing(sqlite3.connect(db_path)) as connection:
                stored = pd.read_sql_query(
                    "SELECT settlement_date, bmu_family_key, dispatch_minus_lost_energy_gap_mwh "
                    "FROM fact_bmu_family_shortfall_daily",
                    connection,
                )
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored.iloc[0]["settlement_date"], "2024-10-01")
            self.assertEqual(stored.iloc[0]["bmu_family_key"], "SGRWO")
            self.assertAlmostEqual(float(stored.iloc[0]["dispatch_minus_lost_energy_gap_mwh"]), 80.0)


if __name__ == "__main__":
    unittest.main()
