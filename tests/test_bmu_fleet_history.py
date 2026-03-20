from __future__ import annotations

import unittest

import pandas as pd

from bmu_fleet_history import build_dim_bmu_fleet_asset


class BmuFleetHistoryTests(unittest.TestCase):
    def test_build_dim_bmu_fleet_asset_keeps_all_production_units_and_maps_parent_region(self) -> None:
        reference = pd.DataFrame(
            [
                {
                    "national_grid_bm_unit": "GAS1",
                    "elexon_bm_unit": "GAS1",
                    "bm_unit_name": "Example Gas",
                    "lead_party_name": "Utility",
                    "fuel_type": "CCGT",
                    "bm_unit_type": "GEN",
                    "production_or_consumption_flag": "P",
                    "gsp_group_id": "_D",
                    "gsp_group_name": "Merseyside North Wales",
                },
                {
                    "national_grid_bm_unit": "SCOT1",
                    "elexon_bm_unit": "SCOT1",
                    "bm_unit_name": "Example Scottish Gas",
                    "lead_party_name": "Utility",
                    "fuel_type": "OCGT",
                    "bm_unit_type": "GEN",
                    "production_or_consumption_flag": "P",
                    "gsp_group_id": "_N",
                    "gsp_group_name": "South Scotland",
                },
                {
                    "national_grid_bm_unit": "HOWAO-1",
                    "elexon_bm_unit": "HOWAO-1",
                    "bm_unit_name": "Hornsea One",
                    "lead_party_name": "Wind Utility",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "production_or_consumption_flag": "P",
                    "gsp_group_id": "_D",
                    "gsp_group_name": "Merseyside North Wales",
                },
                {
                    "national_grid_bm_unit": "LOAD1",
                    "elexon_bm_unit": "LOAD1",
                    "bm_unit_name": "Load",
                    "lead_party_name": "Demand Utility",
                    "fuel_type": "DEMAND",
                    "bm_unit_type": "DEM",
                    "production_or_consumption_flag": "C",
                    "gsp_group_id": "_D",
                    "gsp_group_name": "Merseyside North Wales",
                },
            ]
        )

        fact = build_dim_bmu_fleet_asset(reference)

        self.assertEqual(set(fact["elexon_bm_unit"]), {"GAS1", "SCOT1", "HOWAO-1"})
        gas = fact[fact["elexon_bm_unit"].eq("GAS1")].iloc[0]
        scot = fact[fact["elexon_bm_unit"].eq("SCOT1")].iloc[0]
        hornsea = fact[fact["elexon_bm_unit"].eq("HOWAO-1")].iloc[0]

        self.assertEqual(gas["parent_region"], "England/Wales")
        self.assertEqual(gas["mapping_status"], "region_only")
        self.assertEqual(scot["parent_region"], "Scotland")
        self.assertEqual(scot["mapping_status"], "region_only")
        self.assertEqual(hornsea["cluster_key"], "dogger_hornsea_offshore")
        self.assertEqual(hornsea["parent_region"], "England/Wales")
        self.assertEqual(hornsea["mapping_status"], "mapped")


if __name__ == "__main__":
    unittest.main()
