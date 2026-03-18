import tempfile
import unittest
from pathlib import Path

import pandas as pd

from inline_arbitrage_live import load_system_balance_market_state_input


class InlineArbitrageLiveTests(unittest.TestCase):
    def test_load_system_balance_market_state_input_parses_cached_csv(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2024-11-15",
                    "interval_start_local": "2024-11-15 00:00:00+00:00",
                    "interval_end_local": "2024-11-15 01:00:00+00:00",
                    "interval_start_utc": "2024-11-15T00:00:00Z",
                    "interval_end_utc": "2024-11-15T01:00:00Z",
                    "system_balance_source_published_utc": "2024-11-14T23:45:00Z",
                    "system_balance_feed_available_flag": True,
                    "system_balance_known_flag": False,
                    "system_balance_active_flag": True,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fact_system_balance_market_state_hourly.csv"
            frame.to_csv(path, index=False)
            loaded = load_system_balance_market_state_input(path)

        row = loaded.iloc[0]
        self.assertEqual(row["interval_start_utc"], pd.Timestamp("2024-11-15T00:00:00Z"))
        self.assertEqual(row["system_balance_source_published_utc"], pd.Timestamp("2024-11-14T23:45:00Z"))
        self.assertTrue(bool(row["system_balance_feed_available_flag"]))
        self.assertFalse(bool(row["system_balance_known_flag"]))
        self.assertTrue(bool(row["system_balance_active_flag"]))


if __name__ == "__main__":
    unittest.main()
