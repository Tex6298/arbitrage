from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_supported_range_monthly_backfill import (
    iter_months,
    iter_reviewed_bundle_windows,
    output_complete,
)


class MaterializeSupportedRangeMonthlyBackfillTests(unittest.TestCase):
    def test_iter_months_builds_full_month_windows(self) -> None:
        windows = iter_months(dt.date(2024, 10, 1), dt.date(2024, 12, 1))
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].start_date, dt.date(2024, 10, 1))
        self.assertEqual(windows[0].end_date, dt.date(2024, 10, 31))
        self.assertEqual(windows[1].start_date, dt.date(2024, 11, 1))
        self.assertEqual(windows[1].end_date, dt.date(2024, 11, 30))
        self.assertEqual(windows[2].start_date, dt.date(2024, 12, 1))
        self.assertEqual(windows[2].end_date, dt.date(2024, 12, 31))

    def test_iter_reviewed_bundle_windows_tiles_months_into_supported_chunks(self) -> None:
        month_windows = iter_months(dt.date(2024, 10, 1), dt.date(2024, 10, 1))
        reviewed_windows = iter_reviewed_bundle_windows(month_windows, max_window_days=7)
        self.assertEqual(
            [(window.start_date, window.end_date) for window in reviewed_windows],
            [
                (dt.date(2024, 10, 1), dt.date(2024, 10, 7)),
                (dt.date(2024, 10, 8), dt.date(2024, 10, 14)),
                (dt.date(2024, 10, 15), dt.date(2024, 10, 21)),
                (dt.date(2024, 10, 22), dt.date(2024, 10, 26)),
                (dt.date(2024, 10, 27), dt.date(2024, 10, 27)),
                (dt.date(2024, 10, 28), dt.date(2024, 10, 28)),
                (dt.date(2024, 10, 29), dt.date(2024, 10, 31)),
            ],
        )

    def test_iter_reviewed_bundle_windows_splits_fallback_chunk_into_single_day_tail(self) -> None:
        month_windows = iter_months(dt.date(2025, 10, 1), dt.date(2025, 10, 1))
        reviewed_windows = iter_reviewed_bundle_windows(month_windows, max_window_days=7)
        self.assertEqual(
            [(window.start_date, window.end_date) for window in reviewed_windows],
            [
                (dt.date(2025, 10, 1), dt.date(2025, 10, 7)),
                (dt.date(2025, 10, 8), dt.date(2025, 10, 14)),
                (dt.date(2025, 10, 15), dt.date(2025, 10, 21)),
                (dt.date(2025, 10, 22), dt.date(2025, 10, 25)),
                (dt.date(2025, 10, 26), dt.date(2025, 10, 26)),
                (dt.date(2025, 10, 27), dt.date(2025, 10, 27)),
                (dt.date(2025, 10, 28), dt.date(2025, 10, 28)),
                (dt.date(2025, 10, 29), dt.date(2025, 10, 31)),
            ],
        )

    def test_output_complete_requires_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_dir.mkdir(exist_ok=True)
            self.assertFalse(output_complete(output_dir, "fact_curtailment_opportunity_hourly.csv"))
            (output_dir / "fact_curtailment_opportunity_hourly.csv").write_text("ok", encoding="utf-8")
            self.assertTrue(output_complete(output_dir, "fact_curtailment_opportunity_hourly.csv"))


if __name__ == "__main__":
    unittest.main()
