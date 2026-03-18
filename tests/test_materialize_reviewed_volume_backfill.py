from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_reviewed_volume_backfill import (
    build_monthly_windows,
    copy_existing_bundles,
)


class MaterializeReviewedVolumeBackfillTests(unittest.TestCase):
    def test_build_monthly_windows_uses_anchor_day(self) -> None:
        windows = build_monthly_windows(dt.date(2025, 2, 1), dt.date(2025, 4, 1), anchor_day=15, window_days=3)
        self.assertEqual(
            [(window.start_date.isoformat(), window.end_date.isoformat()) for window in windows],
            [
                ("2025-02-15", "2025-02-17"),
                ("2025-03-15", "2025-03-17"),
                ("2025-04-15", "2025-04-17"),
            ],
        )

    def test_build_monthly_windows_clips_to_month_end(self) -> None:
        windows = build_monthly_windows(dt.date(2025, 2, 1), dt.date(2025, 2, 1), anchor_day=28, window_days=3)
        self.assertEqual(windows[0].start_date.isoformat(), "2025-02-26")
        self.assertEqual(windows[0].end_date.isoformat(), "2025-02-28")

    def test_copy_existing_bundles_copies_matching_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_bundle = root / "curtailment_opportunity_live_britned_reviewed_2025-02-15_2025-02-17"
            source_bundle.mkdir()
            (source_bundle / "fact_curtailment_opportunity_hourly.csv").write_text("header\n", encoding="utf-8")
            (root / "other_dir").mkdir()
            destination_root = root / "bundles"

            copied = copy_existing_bundles(root, destination_root)

            self.assertEqual([path.name for path in copied], [source_bundle.name])
            self.assertTrue((destination_root / source_bundle.name / "fact_curtailment_opportunity_hourly.csv").exists())


if __name__ == "__main__":
    unittest.main()
