import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

import interconnector_itl
from interconnector_itl import ITLDatasetSpec, build_fact_interconnector_itl_hourly


class InterconnectorITLTests(unittest.TestCase):
    def test_build_fact_interconnector_itl_hourly_parses_current_datastore_schema(self) -> None:
        spec = ITLDatasetSpec(
            connector_key="ifa",
            connector_label="IFA",
            border_key="GB-FR",
            target_zone="FR",
            neighbor_domain_key="FR",
            dataset_key="ifa",
            dataset_id="ifa-dataset",
            current_resource_name="IFA ITL Data",
            archived_resource_name="Archived IFA DA & ID Weekly ITLs",
            parse_mode="neso_current_itl",
        )
        resources = [
            {
                "id": "current-resource",
                "name": "IFA ITL Data",
                "url": "https://example.com/ifa.csv",
                "metadata_modified": "2024-10-01T09:30:00Z",
            }
        ]
        current = pd.DataFrame(
            [
                {
                    "Data Upload Time (GMT)": "2024-10-01T10:30:00",
                    "Auction Type": "Intraday 1",
                    "Operational Period Start Date & Time (GMT)": "2024-10-01T10:00:00",
                    "Flow (MW) To GB": 1800,
                    "Reason For Restriction To GB": "No Restriction",
                    "Flow (MW) From GB": 1500,
                    "Reason For Restriction From GB": "Restricted Export",
                }
            ]
        )

        with patch.object(interconnector_itl, "ITL_DATASET_SPECS", (spec,)):
            with patch("interconnector_itl._resource_rows", return_value=resources):
                with patch("interconnector_itl._fetch_csv", return_value=current):
                    fact = build_fact_interconnector_itl_hourly(
                        start_date=dt.date(2024, 10, 1),
                        end_date=dt.date(2024, 10, 1),
                    )

        self.assertEqual(len(fact), 2)
        export_row = fact[fact["direction_key"] == "gb_to_neighbor"].iloc[0]
        self.assertEqual(export_row["connector_key"], "ifa")
        self.assertEqual(export_row["auction_type"], "Intraday 1")
        self.assertAlmostEqual(float(export_row["itl_mw"]), 1500.0)
        self.assertEqual(export_row["restriction_reason"], "Restricted Export")
        self.assertEqual(export_row["itl_state"], "published_restriction")

    def test_build_fact_interconnector_itl_hourly_parses_britned_weekly_archive(self) -> None:
        spec = ITLDatasetSpec(
            connector_key="britned",
            connector_label="BritNed",
            border_key="GB-NL",
            target_zone="NL",
            neighbor_domain_key="NL",
            dataset_key="britned",
            dataset_id="britned-dataset",
            current_resource_name=None,
            archived_resource_name="BritNed DA & ID Weekly ITLs",
            parse_mode="britned_weekly_itl",
        )
        resource = {
            "id": "britned-week",
            "name": "BritNed DA & ID Weekly ITLs 20241002",
            "url": "https://example.com/britned-20241002.csv",
            "metadata_modified": "2024-10-02T09:00:00Z",
        }
        weekly = pd.DataFrame(
            [
                {
                    "Operational Date (YYYY-MM-DD) & Time GMT/BST (HH:MM - HH:MM)": "20241002 23:00 - 00:00",
                    "Flow (MW) From GB": 1000,
                    "Flow (MW) To GB": 950,
                    "Reason For Restiction": "System Security",
                }
            ]
        )

        with patch.object(interconnector_itl, "ITL_DATASET_SPECS", (spec,)):
            with patch("interconnector_itl._resource_rows", return_value=[resource]):
                with patch("interconnector_itl._fetch_csv", return_value=weekly):
                    fact = build_fact_interconnector_itl_hourly(
                        start_date=dt.date(2024, 10, 2),
                        end_date=dt.date(2024, 10, 2),
                    )

        self.assertEqual(len(fact), 2)
        export_row = fact[fact["direction_key"] == "gb_to_neighbor"].iloc[0]
        self.assertEqual(export_row["source_table_variant"], "archived_upload")
        self.assertEqual(export_row["auction_type"], "weekly_archive")
        self.assertAlmostEqual(float(export_row["itl_mw"]), 1000.0)
        self.assertEqual(export_row["interval_end_local"], pd.Timestamp("2024-10-03T00:00:00+01:00"))


if __name__ == "__main__":
    unittest.main()
