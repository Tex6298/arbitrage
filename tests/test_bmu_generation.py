import unittest

import pandas as pd

from bmu_generation import build_dim_bmu_asset


class BMUGenerationTests(unittest.TestCase):
    def test_build_dim_bmu_asset_maps_priority_coverage_families(self) -> None:
        reference = pd.DataFrame(
            [
                {
                    "national_grid_bm_unit": "GNFSW-1",
                    "elexon_bm_unit": "T_GNFSW-1",
                    "bm_unit_name": "Gunfleet Sands 1",
                    "lead_party_name": "Gunfleet Sands Limited",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "production_or_consumption_flag": "P",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test",
                    "generation_capacity_mw": 100.0,
                },
                {
                    "national_grid_bm_unit": "RCBKO-1",
                    "elexon_bm_unit": "T_RCBKO-1",
                    "bm_unit_name": "RACE_BANK_Z01",
                    "lead_party_name": "Race Bank Wind Farm Ltd",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "production_or_consumption_flag": "P",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test",
                    "generation_capacity_mw": 100.0,
                },
                {
                    "national_grid_bm_unit": "CLDNW-1",
                    "elexon_bm_unit": "T_CLDNW-1",
                    "bm_unit_name": "Clyde North",
                    "lead_party_name": "Clyde Windfarm (Scotland) Ltd",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "production_or_consumption_flag": "P",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test",
                    "generation_capacity_mw": 100.0,
                },
                {
                    "national_grid_bm_unit": "PNYCW-1",
                    "elexon_bm_unit": "T_PNYCW-1",
                    "bm_unit_name": "PNYCW-1 Export",
                    "lead_party_name": "Pen y Cymoedd Wind Farm LTD",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "production_or_consumption_flag": "P",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test",
                    "generation_capacity_mw": 100.0,
                },
            ]
        )

        dim = build_dim_bmu_asset(reference).set_index("national_grid_bm_unit")

        self.assertEqual(dim.loc["GNFSW-1", "mapping_status"], "mapped")
        self.assertEqual(dim.loc["GNFSW-1", "cluster_key"], "east_anglia_offshore")
        self.assertEqual(dim.loc["GNFSW-1", "parent_region"], "England/Wales")

        self.assertEqual(dim.loc["RCBKO-1", "mapping_status"], "mapped")
        self.assertEqual(dim.loc["RCBKO-1", "cluster_key"], "humber_offshore")
        self.assertEqual(dim.loc["RCBKO-1", "parent_region"], "England/Wales")

        self.assertEqual(dim.loc["CLDNW-1", "mapping_status"], "region_only")
        self.assertTrue(pd.isna(dim.loc["CLDNW-1", "cluster_key"]))
        self.assertEqual(dim.loc["CLDNW-1", "parent_region"], "Scotland")

        self.assertEqual(dim.loc["PNYCW-1", "mapping_status"], "region_only")
        self.assertTrue(pd.isna(dim.loc["PNYCW-1", "cluster_key"]))
        self.assertEqual(dim.loc["PNYCW-1", "parent_region"], "England/Wales")


if __name__ == "__main__":
    unittest.main()
