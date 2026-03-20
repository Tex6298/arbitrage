from __future__ import annotations

import unittest

import pandas as pd

from reviewed_opportunity_displacement import (
    OPPORTUNITY_DISPLACEMENT_DAILY_TABLE,
    OPPORTUNITY_DISPLACEMENT_FUEL_HOURLY_TABLE,
    OPPORTUNITY_DISPLACEMENT_HOURLY_TABLE,
    build_fact_fossil_stack_hourly,
    build_fact_opportunity_displacement,
)


class ReviewedOpportunityDisplacementTests(unittest.TestCase):
    def test_build_fact_fossil_stack_hourly_and_allocate_same_region_generation(self) -> None:
        generation = pd.DataFrame(
            [
                {
                    "half_hour_start_time_utc": "2024-10-01T00:00:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "generation_mwh": 20.0,
                },
                {
                    "half_hour_start_time_utc": "2024-10-01T00:30:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "generation_mwh": 20.0,
                },
                {
                    "half_hour_start_time_utc": "2024-10-01T00:00:00Z",
                    "elexon_bm_unit": "COAL1",
                    "fuel_type": "COAL",
                    "parent_region": "England/Wales",
                    "generation_mwh": 10.0,
                },
                {
                    "half_hour_start_time_utc": "2024-10-01T00:30:00Z",
                    "elexon_bm_unit": "COAL1",
                    "fuel_type": "COAL",
                    "parent_region": "England/Wales",
                    "generation_mwh": 10.0,
                },
                {
                    "half_hour_start_time_utc": "2024-10-01T00:00:00Z",
                    "elexon_bm_unit": "WIND1",
                    "fuel_type": "WIND",
                    "parent_region": "England/Wales",
                    "generation_mwh": 50.0,
                },
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "availability_state": "available",
                    "remit_max_available_capacity_mw": 50.0,
                    "generation_capacity_mw": 60.0,
                },
                {
                    "interval_start_utc": "2024-10-01T00:30:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "availability_state": "available",
                    "remit_max_available_capacity_mw": 50.0,
                    "generation_capacity_mw": 60.0,
                },
            ]
        )
        bid_offer = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "minimum_offer_gbp_per_mwh": 70.0,
                    "maximum_offer_gbp_per_mwh": 100.0,
                },
                {
                    "interval_start_utc": "2024-10-01T00:30:00Z",
                    "elexon_bm_unit": "COAL1",
                    "fuel_type": "COAL",
                    "parent_region": "England/Wales",
                    "minimum_offer_gbp_per_mwh": 90.0,
                    "maximum_offer_gbp_per_mwh": 120.0,
                },
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "accepted_down_delta_mwh_lower_bound": 2.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                }
            ]
        )
        opportunity = pd.DataFrame(
            [
                {
                    "date": "2024-10-01",
                    "interval_start_utc": "2024-10-01T00:00:00Z",
                    "cluster_key": "dogger_hornsea_offshore",
                    "cluster_label": "Dogger and Hornsea Offshore",
                    "parent_region": "England/Wales",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "route_label": "GB-NL",
                    "route_border_key": "GB-NL",
                    "route_target_zone": "NL",
                    "hub_key": "britned",
                    "hub_label": "BritNed",
                    "opportunity_deliverable_mwh": 30.0,
                    "opportunity_gross_value_eur": 1500.0,
                }
            ]
        )
        factors = pd.DataFrame(
            [
                {"fuel_type": "CCGT", "emission_factor_tco2_per_mwh": 0.4},
                {"fuel_type": "COAL", "emission_factor_tco2_per_mwh": 0.9},
            ]
        )

        fossil_stack = build_fact_fossil_stack_hourly(
            generation,
            fact_bmu_availability_half_hourly=availability,
            fact_bmu_bid_offer_half_hourly=bid_offer,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
        )
        frames = build_fact_opportunity_displacement(opportunity, fossil_stack, emission_factors=factors)

        hourly = frames[OPPORTUNITY_DISPLACEMENT_HOURLY_TABLE]
        fuel_hourly = frames[OPPORTUNITY_DISPLACEMENT_FUEL_HOURLY_TABLE]
        daily = frames[OPPORTUNITY_DISPLACEMENT_DAILY_TABLE]

        self.assertEqual(len(hourly), 1)
        self.assertAlmostEqual(float(hourly.iloc[0]["displaced_fossil_mwh"]), 30.0)
        self.assertEqual(hourly.iloc[0]["displacement_allocation_basis"], "physical_generation")

        by_fuel = {
            row["fuel_type"]: float(row["allocated_displaced_fossil_mwh"])
            for _, row in fuel_hourly.iterrows()
        }
        self.assertAlmostEqual(by_fuel["CCGT"], 20.0)
        self.assertAlmostEqual(by_fuel["COAL"], 10.0)
        self.assertAlmostEqual(float(daily["displaced_fossil_mwh"].sum()), 30.0)
        self.assertAlmostEqual(float(fuel_hourly["allocated_displaced_emissions_tco2"].sum()), 17.0)

    def test_build_fact_opportunity_displacement_falls_back_to_available_capacity(self) -> None:
        generation = pd.DataFrame(
            [
                {
                    "half_hour_start_time_utc": "2024-10-01T01:00:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "generation_mwh": 0.0,
                }
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "interval_start_utc": "2024-10-01T01:00:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "availability_state": "available",
                    "remit_max_available_capacity_mw": 20.0,
                    "generation_capacity_mw": 20.0,
                },
                {
                    "interval_start_utc": "2024-10-01T01:30:00Z",
                    "elexon_bm_unit": "GAS1",
                    "fuel_type": "CCGT",
                    "parent_region": "England/Wales",
                    "availability_state": "available",
                    "remit_max_available_capacity_mw": 20.0,
                    "generation_capacity_mw": 20.0,
                },
            ]
        )
        opportunity = pd.DataFrame(
            [
                {
                    "date": "2024-10-01",
                    "interval_start_utc": "2024-10-01T01:00:00Z",
                    "cluster_key": "dogger_hornsea_offshore",
                    "parent_region": "England/Wales",
                    "route_name": "R2_netback_GB_NL_DE_PL",
                    "route_target_zone": "NL",
                    "hub_key": "britned",
                    "opportunity_deliverable_mwh": 8.0,
                    "opportunity_gross_value_eur": 400.0,
                }
            ]
        )

        fossil_stack = build_fact_fossil_stack_hourly(
            generation,
            fact_bmu_availability_half_hourly=availability,
        )
        frames = build_fact_opportunity_displacement(opportunity, fossil_stack)
        hourly = frames[OPPORTUNITY_DISPLACEMENT_HOURLY_TABLE]

        self.assertEqual(hourly.iloc[0]["displacement_allocation_basis"], "available_capacity")
        self.assertAlmostEqual(float(hourly.iloc[0]["displaced_fossil_mwh"]), 8.0)


if __name__ == "__main__":
    unittest.main()
