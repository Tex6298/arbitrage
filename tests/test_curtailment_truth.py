import datetime as dt
import unittest

import numpy as np
import pandas as pd

from bmu_availability import build_fact_bmu_availability_half_hourly
from bmu_dispatch import build_fact_bmu_dispatch_acceptance_half_hourly
from bmu_physical import build_fact_bmu_physical_position_half_hourly
from bmu_truth_utils import build_half_hour_interval_frame
from curtailment_truth import build_fact_bmu_curtailment_truth_half_hourly, filter_truth_profile
from weather_history import build_fact_weather_hourly_from_anchor_weather


def sample_dim_bmu_asset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "elexon_bm_unit": "T_TEST-1",
                "national_grid_bm_unit": "TEST-1",
                "bm_unit_name": "Test Wind 1",
                "lead_party_name": "Test Lead",
                "fuel_type": "WIND",
                "bm_unit_type": "GEN",
                "gsp_group_id": "_A",
                "gsp_group_name": "Test GSP",
                "generation_capacity_mw": 20.0,
                "mapping_status": "mapped",
                "mapping_confidence": "high",
                "mapping_rule": "test",
                "cluster_key": "moray_firth_offshore",
                "cluster_label": "Moray Firth Offshore",
                "parent_region": "Scotland",
            }
        ]
    )


class CurtailmentTruthTests(unittest.TestCase):
    def test_half_hour_spine_handles_dst_days(self) -> None:
        spring = build_half_hour_interval_frame(dt.date(2026, 3, 29), dt.date(2026, 3, 29))
        autumn = build_half_hour_interval_frame(dt.date(2026, 10, 25), dt.date(2026, 10, 25))
        normal = build_half_hour_interval_frame(dt.date(2026, 3, 10), dt.date(2026, 3, 10))
        self.assertEqual(len(normal), 48)
        self.assertEqual(len(spring), 46)
        self.assertEqual(len(autumn), 50)

    def test_overlapping_acceptances_aggregate_into_one_half_hour(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period_from": 1,
                    "settlement_period_to": 1,
                    "time_from_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "time_to_utc": pd.Timestamp("2024-09-30T23:20:00Z"),
                    "time_from_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "time_to_local": pd.Timestamp("2024-10-01T00:20:00+01:00"),
                    "acceptance_time_utc": pd.Timestamp("2024-09-30T22:55:00Z"),
                    "acceptance_time_local": pd.Timestamp("2024-09-30T23:55:00+01:00"),
                    "duration_hours": 20.0 / 60.0,
                    "duration_minutes": 20.0,
                    "source_key": "BOALF",
                    "source_label": "test",
                    "source_dataset": "BOALF",
                    "target_is_proxy": False,
                    "dispatch_truth_tier": "dispatch_acceptance_lower_bound",
                    "is_lower_bound_metric": True,
                    "acceptance_number": 1,
                    "amendment_flag": False,
                    "deemed_bo_flag": False,
                    "so_flag": False,
                    "stor_flag": False,
                    "rr_flag": False,
                    "elexon_bm_unit": "T_TEST-1",
                    "national_grid_bm_unit": "TEST-1",
                    "national_grid_bm_unit_from_fact": "TEST-1",
                    "bm_unit_name": "Test Wind 1",
                    "lead_party_name": "Test Lead",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test GSP",
                    "generation_capacity_mw": 20.0,
                    "level_from_mw": 10.0,
                    "level_to_mw": 6.0,
                    "level_delta_mw": -4.0,
                    "accepted_level_mean_mw": 8.0,
                    "accepted_down_delta_mw": 4.0,
                    "accepted_up_delta_mw": 0.0,
                    "accepted_down_delta_mwh_lower_bound": 4.0 * (20.0 / 60.0),
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_direction": "down",
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period_from": 1,
                    "settlement_period_to": 1,
                    "time_from_utc": pd.Timestamp("2024-09-30T23:10:00Z"),
                    "time_to_utc": pd.Timestamp("2024-09-30T23:30:00Z"),
                    "time_from_local": pd.Timestamp("2024-10-01T00:10:00+01:00"),
                    "time_to_local": pd.Timestamp("2024-10-01T00:30:00+01:00"),
                    "acceptance_time_utc": pd.Timestamp("2024-09-30T23:05:00Z"),
                    "acceptance_time_local": pd.Timestamp("2024-10-01T00:05:00+01:00"),
                    "duration_hours": 20.0 / 60.0,
                    "duration_minutes": 20.0,
                    "source_key": "BOALF",
                    "source_label": "test",
                    "source_dataset": "BOALF",
                    "target_is_proxy": False,
                    "dispatch_truth_tier": "dispatch_acceptance_lower_bound",
                    "is_lower_bound_metric": True,
                    "acceptance_number": 2,
                    "amendment_flag": False,
                    "deemed_bo_flag": False,
                    "so_flag": False,
                    "stor_flag": False,
                    "rr_flag": False,
                    "elexon_bm_unit": "T_TEST-1",
                    "national_grid_bm_unit": "TEST-1",
                    "national_grid_bm_unit_from_fact": "TEST-1",
                    "bm_unit_name": "Test Wind 1",
                    "lead_party_name": "Test Lead",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test GSP",
                    "generation_capacity_mw": 20.0,
                    "level_from_mw": 6.0,
                    "level_to_mw": 4.0,
                    "level_delta_mw": -2.0,
                    "accepted_level_mean_mw": 5.0,
                    "accepted_down_delta_mw": 2.0,
                    "accepted_up_delta_mw": 0.0,
                    "accepted_down_delta_mwh_lower_bound": 2.0 * (20.0 / 60.0),
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_direction": "down",
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                },
            ]
        )
        fact = build_fact_bmu_dispatch_acceptance_half_hourly(events, dt.date(2024, 10, 1), dt.date(2024, 10, 1))
        first_row = fact.iloc[0]
        self.assertEqual(first_row["acceptance_event_count"], 2)
        self.assertEqual(first_row["distinct_acceptance_number_count"], 2)
        self.assertAlmostEqual(first_row["accepted_down_delta_mwh_lower_bound"], 2.0)

    def test_physical_baseline_below_generation_is_invalid(self) -> None:
        dim = sample_dim_bmu_asset()
        generation = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "half_hour_start_time_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "half_hour_end_time_local": pd.Timestamp("2024-10-01T00:30:00+01:00"),
                    "half_hour_start_time_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "half_hour_end_time_utc": pd.Timestamp("2024-09-30T23:30:00Z"),
                    "source_key": "B1610",
                    "source_label": "test",
                    "source_dataset": "B1610",
                    "target_is_proxy": False,
                    "elexon_bm_unit": "T_TEST-1",
                    "national_grid_bm_unit": "TEST-1",
                    "national_grid_bm_unit_from_fact": "TEST-1",
                    "bm_unit_name": "Test Wind 1",
                    "lead_party_name": "Test Lead",
                    "fuel_type": "WIND",
                    "bm_unit_type": "GEN",
                    "gsp_group_id": "_A",
                    "gsp_group_name": "Test GSP",
                    "generation_capacity_mw": 20.0,
                    "generation_mwh": 10.0,
                    "generation_mw_equivalent": 20.0,
                    "capacity_factor": 1.0,
                    "mapping_status": "mapped",
                    "mapping_confidence": "high",
                    "mapping_rule": "test",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                }
            ]
        )
        raw_physical = pd.DataFrame(
            [
                {
                    "dataset": "PN",
                    "settlementDate": "2024-10-01",
                    "settlementPeriod": 1,
                    "timeFrom": "2024-09-30T23:00:00Z",
                    "timeTo": "2024-09-30T23:30:00Z",
                    "levelFrom": 16.0,
                    "levelTo": 16.0,
                    "nationalGridBmUnit": "TEST-1",
                    "bmUnit": "T_TEST-1",
                }
            ]
        )
        fact = build_fact_bmu_physical_position_half_hourly(dim, generation, raw_physical, dt.date(2024, 10, 1), dt.date(2024, 10, 1))
        first_row = fact[fact["settlement_period"] == 1].iloc[0]
        self.assertFalse(bool(first_row["counterfactual_valid_flag"]))

    def test_missing_remit_keeps_availability_unknown(self) -> None:
        dim = sample_dim_bmu_asset()
        raw_uou = pd.DataFrame(
            [
                {
                    "dataset": "UOU2T14D",
                    "bmUnit": "T_TEST-1",
                    "nationalGridBmUnit": "TEST-1",
                    "publishTime": "2026-03-10T00:00:00Z",
                    "forecastDate": "2024-10-01",
                    "outputUsable": 20.0,
                }
            ]
        )
        fact = build_fact_bmu_availability_half_hourly(
            dim_bmu_asset=dim,
            raw_remit_frame=pd.DataFrame(),
            raw_uou_frame=raw_uou,
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
            remit_fetch_ok=False,
        )
        first_row = fact[fact["settlement_period"] == 1].iloc[0]
        self.assertEqual(first_row["availability_state"], "unknown")

    def test_weather_history_aggregates_anchor_weights(self) -> None:
        anchor_weather = pd.DataFrame(
            [
                {
                    "date": dt.date(2024, 10, 1),
                    "hour_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "hour_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "hour_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "hour_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "source_key": "open_meteo_archive",
                    "source_label": "test",
                    "source_dataset": "historical_weather_best_match",
                    "target_is_proxy": False,
                    "anchor_key": "beatrice",
                    "anchor_label": "Beatrice",
                    "requested_latitude": 58.1,
                    "requested_longitude": -2.6,
                    "resolved_latitude": 58.1,
                    "resolved_longitude": -2.6,
                    "resolved_elevation_m": 0.0,
                    "cell_selection": "sea",
                    "temperature_2m_c": 8.0,
                    "pressure_msl_hpa": 1005.0,
                    "cloud_cover_pct": 50.0,
                    "wind_speed_10m_ms": 9.0,
                    "wind_speed_100m_ms": 12.0,
                    "wind_direction_100m_deg": 180.0,
                    "wind_gusts_10m_ms": 12.0,
                    "wind_u_100m_ms": 0.0,
                    "wind_v_100m_ms": 12.0,
                    "wind_power_index_100m": 1728.0,
                    "wind_speed_ratio_100m_to_10m": 12.0 / 9.0,
                },
                {
                    "date": dt.date(2024, 10, 1),
                    "hour_start_local": pd.Timestamp("2024-10-01T00:00:00+01:00"),
                    "hour_end_local": pd.Timestamp("2024-10-01T01:00:00+01:00"),
                    "hour_start_utc": pd.Timestamp("2024-09-30T23:00:00Z"),
                    "hour_end_utc": pd.Timestamp("2024-10-01T00:00:00Z"),
                    "source_key": "open_meteo_archive",
                    "source_label": "test",
                    "source_dataset": "historical_weather_best_match",
                    "target_is_proxy": False,
                    "anchor_key": "moray_east",
                    "anchor_label": "Moray East",
                    "requested_latitude": 57.7,
                    "requested_longitude": -2.3,
                    "resolved_latitude": 57.7,
                    "resolved_longitude": -2.3,
                    "resolved_elevation_m": 0.0,
                    "cell_selection": "sea",
                    "temperature_2m_c": 10.0,
                    "pressure_msl_hpa": 1007.0,
                    "cloud_cover_pct": 70.0,
                    "wind_speed_10m_ms": 11.0,
                    "wind_speed_100m_ms": 14.0,
                    "wind_direction_100m_deg": 180.0,
                    "wind_gusts_10m_ms": 14.0,
                    "wind_u_100m_ms": 0.0,
                    "wind_v_100m_ms": 14.0,
                    "wind_power_index_100m": 2744.0,
                    "wind_speed_ratio_100m_to_10m": 14.0 / 11.0,
                },
            ]
        )
        fact_weather = build_fact_weather_hourly_from_anchor_weather(anchor_weather)
        cluster_row = fact_weather[
            (fact_weather["scope_type"] == "cluster")
            & (fact_weather["scope_key"] == "moray_firth_offshore")
        ].iloc[0]
        expected_wind_speed = (12.0 * 588.0 + 14.0 * 950.0) / (588.0 + 950.0)
        self.assertAlmostEqual(cluster_row["wind_speed_100m_ms"], expected_wind_speed)
        self.assertEqual(int(cluster_row["weather_anchor_count"]), 2)

    def test_weather_curve_can_upgrade_invalid_dispatch_row(self) -> None:
        dim = sample_dim_bmu_asset()
        base_day = dt.date(2024, 10, 1)
        generation_rows = []
        availability_rows = []
        dispatch_rows = []
        weather_rows = []
        for hour in range(24):
            hour_start_utc = pd.Timestamp("2024-09-30T23:00:00Z") + pd.Timedelta(hours=hour)
            hour_start_local = hour_start_utc.tz_convert("Europe/London")
            weather_speed = 4.0 + hour
            weather_rows.append(
                {
                    "date": hour_start_local.date(),
                    "hour_start_local": hour_start_local,
                    "hour_end_local": hour_start_local + pd.Timedelta(hours=1),
                    "hour_start_utc": hour_start_utc,
                    "hour_end_utc": hour_start_utc + pd.Timedelta(hours=1),
                    "scope_type": "cluster",
                    "scope_key": "moray_firth_offshore",
                    "scope_label": "Moray Firth Offshore",
                    "cluster_key": "moray_firth_offshore",
                    "cluster_label": "Moray Firth Offshore",
                    "parent_region": "Scotland",
                    "source_key": "open_meteo_archive",
                    "source_label": "test",
                    "source_dataset": "historical_weather_best_match",
                    "target_is_proxy": False,
                    "weather_anchor_count": 1,
                    "weather_weight_sum_mw": 1000.0,
                    "temperature_2m_c": 8.0,
                    "pressure_msl_hpa": 1000.0,
                    "cloud_cover_pct": 50.0,
                    "wind_speed_10m_ms": weather_speed - 2.0,
                    "wind_speed_100m_ms": weather_speed,
                    "wind_direction_100m_deg": 180.0,
                    "wind_gusts_10m_ms": weather_speed + 1.0,
                    "wind_u_100m_ms": 0.0,
                    "wind_v_100m_ms": weather_speed,
                    "wind_power_index_100m": weather_speed ** 3,
                    "wind_speed_ratio_100m_to_10m": weather_speed / (weather_speed - 2.0),
                    "resolved_latitude": 58.0,
                    "resolved_longitude": -2.0,
                    "resolved_elevation_m": 0.0,
                }
            )
            for half_hour in range(2):
                settlement_period = hour * 2 + half_hour + 1
                interval_start_utc = hour_start_utc + pd.Timedelta(minutes=30 * half_hour)
                generation_mwh = min((weather_speed / 16.0) ** 3 * 10.0, 10.0)
                if settlement_period == 25:
                    generation_mwh = 2.0
                    dispatch_rows.append(
                        {
                            "settlement_date": base_day,
                            "settlement_period": settlement_period,
                            "elexon_bm_unit": "T_TEST-1",
                            "accepted_down_delta_mwh_lower_bound": 3.0,
                            "accepted_up_delta_mwh_lower_bound": 0.0,
                            "dispatch_down_flag": True,
                            "dispatch_up_flag": False,
                            "acceptance_event_count": 1,
                            "distinct_acceptance_number_count": 1,
                        }
                    )
                generation_rows.append(
                    {
                        "settlement_date": base_day,
                        "settlement_period": settlement_period,
                        "elexon_bm_unit": "T_TEST-1",
                        "generation_mwh": generation_mwh,
                    }
                )
                availability_rows.append(
                    {
                        "settlement_date": base_day,
                        "settlement_period": settlement_period,
                        "elexon_bm_unit": "T_TEST-1",
                        "remit_active_flag": False,
                        "availability_state": "available",
                        "availability_confidence": "high",
                        "uou_output_usable_mw": 20.0,
                    }
                )

        physical = pd.DataFrame(
            [
                {
                    "settlement_date": base_day,
                    "settlement_period": 25,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": pd.NA,
                    "physical_baseline_mwh": np.nan,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "none",
                    "counterfactual_valid_flag": False,
                }
            ]
        )
        constraints = pd.DataFrame([{"date": base_day, "total_curtailment_mwh": 999.0}])

        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=pd.DataFrame(generation_rows),
            fact_bmu_dispatch_acceptance_half_hourly=pd.DataFrame(dispatch_rows),
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=pd.DataFrame(availability_rows),
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(weather_rows),
            start_date=base_day,
            end_date=base_day,
        )
        upgraded_row = fact[fact["settlement_period"] == 25].iloc[0]
        self.assertEqual(upgraded_row["truth_tier"], "weather_calibrated")
        self.assertTrue(bool(upgraded_row["lost_energy_estimate_flag"]))
        self.assertEqual(upgraded_row["counterfactual_method"], "bmu_weather_power_curve")

    def test_truth_profiles_follow_row_flags(self) -> None:
        dim = sample_dim_bmu_asset()
        generation = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 4.0,
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "generation_mwh": 5.0,
                },
            ]
        )
        dispatch = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "accepted_down_delta_mwh_lower_bound": 2.0,
                    "accepted_up_delta_mwh_lower_bound": 0.0,
                    "dispatch_down_flag": True,
                    "dispatch_up_flag": False,
                    "acceptance_event_count": 1,
                    "distinct_acceptance_number_count": 1,
                }
            ]
        )
        physical = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 6.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "physical_baseline_source_dataset": "PN",
                    "physical_baseline_mwh": 5.0,
                    "physical_consistency_flag": True,
                    "counterfactual_method": "pn_qpn_physical_max",
                    "counterfactual_valid_flag": True,
                },
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 1,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                },
                {
                    "settlement_date": dt.date(2024, 10, 1),
                    "settlement_period": 2,
                    "elexon_bm_unit": "T_TEST-1",
                    "remit_active_flag": False,
                    "availability_state": "available",
                    "availability_confidence": "high",
                    "uou_output_usable_mw": 20.0,
                },
            ]
        )
        constraints = pd.DataFrame(
            [
                {
                    "date": dt.date(2024, 10, 1),
                    "total_curtailment_mwh": 2.0,
                }
            ]
        )
        fact = build_fact_bmu_curtailment_truth_half_hourly(
            dim_bmu_asset=dim,
            fact_bmu_generation_half_hourly=generation,
            fact_bmu_dispatch_acceptance_half_hourly=dispatch,
            fact_bmu_physical_position_half_hourly=physical,
            fact_bmu_availability_half_hourly=availability,
            fact_constraint_daily=constraints,
            fact_weather_hourly=pd.DataFrame(),
            start_date=dt.date(2024, 10, 1),
            end_date=dt.date(2024, 10, 1),
        )
        period_1 = fact[fact["settlement_period"] == 1].iloc[0]
        period_2 = fact[fact["settlement_period"] == 2].iloc[0]
        self.assertEqual(period_1["truth_tier"], "physical_baseline")
        self.assertTrue(bool(period_1["precision_profile_include"]))
        self.assertTrue(bool(period_1["research_profile_include"]))
        self.assertEqual(period_2["truth_tier"], "physical_baseline")
        self.assertTrue(bool(period_2["precision_profile_include"]))
        self.assertFalse(bool(period_2["research_profile_include"]))

        precision = filter_truth_profile(fact, "precision")
        research = filter_truth_profile(fact, "research")
        self.assertEqual(set(precision["settlement_period"]), {1, 2})
        self.assertEqual(set(research["settlement_period"]), {1})


if __name__ == "__main__":
    unittest.main()
