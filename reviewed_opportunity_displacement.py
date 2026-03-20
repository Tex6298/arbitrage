from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from opportunity_backtest import load_curtailment_opportunity_input


FOSSIL_STACK_HOURLY_TABLE = "fact_fossil_stack_hourly"
OPPORTUNITY_DISPLACEMENT_HOURLY_TABLE = "fact_opportunity_displacement_hourly"
OPPORTUNITY_DISPLACEMENT_FUEL_HOURLY_TABLE = "fact_opportunity_displacement_fuel_hourly"
OPPORTUNITY_DISPLACEMENT_DAILY_TABLE = "fact_opportunity_displacement_daily"

FOSSIL_FUEL_KEYWORDS = ("GAS", "COAL", "OIL", "DIESEL", "OCGT", "CCGT")
NON_FOSSIL_FUELS = {
    "WIND",
    "SOLAR",
    "HYDRO",
    "PS",
    "PSHYD",
    "NPSHYD",
    "NUCLEAR",
    "BIOMASS",
    "BIOGAS",
    "WASTE",
    "OTHER",
    "INTERCONNECTOR",
    "BATTERY",
    "STORAGE",
}


def _normalize_interval_to_hour(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.floor("h")


def _is_fossil_fuel(value: object) -> bool:
    if pd.isna(value):
        return False
    normalized = str(value).strip().upper()
    if not normalized or normalized in NON_FOSSIL_FUELS:
        return False
    if "BIO" in normalized:
        return False
    return any(keyword in normalized for keyword in FOSSIL_FUEL_KEYWORDS)


def _build_hourly_generation(
    fact_bmu_generation_half_hourly: pd.DataFrame,
    *,
    region_column: str,
) -> pd.DataFrame:
    frame = fact_bmu_generation_half_hourly.copy()
    if frame.empty:
        return pd.DataFrame(columns=["interval_start_utc", region_column, "fuel_type", "fossil_generation_mwh", "fossil_bmu_count"])
    frame = frame[frame["fuel_type"].map(_is_fossil_fuel)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["interval_start_utc", region_column, "fuel_type", "fossil_generation_mwh", "fossil_bmu_count"])
    frame["interval_start_utc"] = _normalize_interval_to_hour(frame["half_hour_start_time_utc"])
    frame["generation_mwh"] = pd.to_numeric(frame.get("generation_mwh"), errors="coerce").fillna(0.0)
    frame = frame[frame["interval_start_utc"].notna() & frame[region_column].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["interval_start_utc", region_column, "fuel_type", "fossil_generation_mwh", "fossil_bmu_count"])
    grouped = (
        frame.groupby(["interval_start_utc", region_column, "fuel_type"], dropna=False)
        .agg(
            fossil_generation_mwh=("generation_mwh", "sum"),
            fossil_bmu_count=("elexon_bm_unit", "nunique"),
        )
        .reset_index()
    )
    return grouped


def _build_hourly_availability(
    fact_bmu_availability_half_hourly: pd.DataFrame,
    *,
    region_column: str,
) -> pd.DataFrame:
    columns = ["interval_start_utc", region_column, "fuel_type", "fossil_available_capacity_mwh_upper_bound"]
    frame = fact_bmu_availability_half_hourly.copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame[frame["fuel_type"].map(_is_fossil_fuel)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["interval_start_utc"] = _normalize_interval_to_hour(frame["interval_start_utc"])
    frame["generation_capacity_mw"] = pd.to_numeric(frame.get("generation_capacity_mw"), errors="coerce")
    frame["remit_max_available_capacity_mw"] = pd.to_numeric(frame.get("remit_max_available_capacity_mw"), errors="coerce")
    available_mask = frame.get("availability_state", pd.Series(pd.NA, index=frame.index)).astype("string").eq("available")
    frame["available_capacity_mwh_half_hour"] = 0.0
    frame.loc[available_mask, "available_capacity_mwh_half_hour"] = (
        frame.loc[available_mask, "remit_max_available_capacity_mw"]
        .where(frame.loc[available_mask, "remit_max_available_capacity_mw"].notna(), frame.loc[available_mask, "generation_capacity_mw"])
        .fillna(0.0)
        * 0.5
    )
    frame = frame[frame["interval_start_utc"].notna() & frame[region_column].notna()].copy()
    grouped = (
        frame.groupby(["interval_start_utc", region_column, "fuel_type"], dropna=False)
        .agg(fossil_available_capacity_mwh_upper_bound=("available_capacity_mwh_half_hour", "sum"))
        .reset_index()
    )
    return grouped


def _build_hourly_bid_offer(
    fact_bmu_bid_offer_half_hourly: pd.DataFrame,
    *,
    region_column: str,
) -> pd.DataFrame:
    columns = ["interval_start_utc", region_column, "fuel_type", "fossil_offer_unit_count", "minimum_offer_gbp_per_mwh", "maximum_offer_gbp_per_mwh"]
    frame = fact_bmu_bid_offer_half_hourly.copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame[frame["fuel_type"].map(_is_fossil_fuel)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["interval_start_utc"] = _normalize_interval_to_hour(frame["interval_start_utc"])
    frame["minimum_offer_gbp_per_mwh"] = pd.to_numeric(frame.get("minimum_offer_gbp_per_mwh"), errors="coerce")
    frame["maximum_offer_gbp_per_mwh"] = pd.to_numeric(frame.get("maximum_offer_gbp_per_mwh"), errors="coerce")
    offer_mask = frame["minimum_offer_gbp_per_mwh"].notna() | frame["maximum_offer_gbp_per_mwh"].notna()
    frame = frame[frame["interval_start_utc"].notna() & frame[region_column].notna()].copy()
    grouped = (
        frame.groupby(["interval_start_utc", region_column, "fuel_type"], dropna=False)
        .agg(
            fossil_offer_unit_count=("elexon_bm_unit", lambda values: int(pd.Series(values)[offer_mask.loc[values.index]].nunique())),
            minimum_offer_gbp_per_mwh=("minimum_offer_gbp_per_mwh", "min"),
            maximum_offer_gbp_per_mwh=("maximum_offer_gbp_per_mwh", "max"),
        )
        .reset_index()
    )
    return grouped


def _build_hourly_dispatch(
    fact_bmu_dispatch_acceptance_half_hourly: pd.DataFrame,
    *,
    region_column: str,
) -> pd.DataFrame:
    columns = [
        "interval_start_utc",
        region_column,
        "fuel_type",
        "fossil_dispatch_down_mwh_lower_bound",
        "fossil_dispatch_up_mwh_lower_bound",
    ]
    frame = fact_bmu_dispatch_acceptance_half_hourly.copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame[frame["fuel_type"].map(_is_fossil_fuel)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["interval_start_utc"] = _normalize_interval_to_hour(frame["interval_start_utc"])
    frame["accepted_down_delta_mwh_lower_bound"] = pd.to_numeric(
        frame.get("accepted_down_delta_mwh_lower_bound"), errors="coerce"
    ).fillna(0.0)
    frame["accepted_up_delta_mwh_lower_bound"] = pd.to_numeric(
        frame.get("accepted_up_delta_mwh_lower_bound"), errors="coerce"
    ).fillna(0.0)
    frame = frame[frame["interval_start_utc"].notna() & frame[region_column].notna()].copy()
    grouped = (
        frame.groupby(["interval_start_utc", region_column, "fuel_type"], dropna=False)
        .agg(
            fossil_dispatch_down_mwh_lower_bound=("accepted_down_delta_mwh_lower_bound", "sum"),
            fossil_dispatch_up_mwh_lower_bound=("accepted_up_delta_mwh_lower_bound", "sum"),
        )
        .reset_index()
    )
    return grouped


def build_fact_fossil_stack_hourly(
    fact_bmu_generation_half_hourly: pd.DataFrame,
    *,
    fact_bmu_availability_half_hourly: pd.DataFrame | None = None,
    fact_bmu_bid_offer_half_hourly: pd.DataFrame | None = None,
    fact_bmu_dispatch_acceptance_half_hourly: pd.DataFrame | None = None,
    region_column: str = "parent_region",
) -> pd.DataFrame:
    generation = _build_hourly_generation(fact_bmu_generation_half_hourly, region_column=region_column)
    availability = _build_hourly_availability(
        fact_bmu_availability_half_hourly if fact_bmu_availability_half_hourly is not None else pd.DataFrame(),
        region_column=region_column,
    )
    bid_offer = _build_hourly_bid_offer(
        fact_bmu_bid_offer_half_hourly if fact_bmu_bid_offer_half_hourly is not None else pd.DataFrame(),
        region_column=region_column,
    )
    dispatch = _build_hourly_dispatch(
        fact_bmu_dispatch_acceptance_half_hourly
        if fact_bmu_dispatch_acceptance_half_hourly is not None
        else pd.DataFrame(),
        region_column=region_column,
    )

    stack = generation.copy()
    for frame in (availability, bid_offer, dispatch):
        if stack.empty:
            stack = frame.copy()
        elif not frame.empty:
            stack = stack.merge(frame, on=["interval_start_utc", region_column, "fuel_type"], how="outer")
    if stack.empty:
        return pd.DataFrame(
            columns=[
                "interval_start_utc",
                region_column,
                "fuel_type",
                "fossil_generation_mwh",
                "fossil_bmu_count",
                "fossil_available_capacity_mwh_upper_bound",
                "fossil_offer_unit_count",
                "minimum_offer_gbp_per_mwh",
                "maximum_offer_gbp_per_mwh",
                "fossil_dispatch_down_mwh_lower_bound",
                "fossil_dispatch_up_mwh_lower_bound",
                "source_lineage",
            ]
        )

    numeric_columns = [
        "fossil_generation_mwh",
        "fossil_bmu_count",
        "fossil_available_capacity_mwh_upper_bound",
        "fossil_offer_unit_count",
        "minimum_offer_gbp_per_mwh",
        "maximum_offer_gbp_per_mwh",
        "fossil_dispatch_down_mwh_lower_bound",
        "fossil_dispatch_up_mwh_lower_bound",
    ]
    for column in numeric_columns:
        stack[column] = pd.to_numeric(stack.get(column), errors="coerce")
    stack["source_lineage"] = (
        "fact_bmu_generation_half_hourly|fact_bmu_availability_half_hourly|"
        "fact_bmu_bid_offer_half_hourly|fact_bmu_dispatch_acceptance_half_hourly"
    )
    return stack.sort_values(["interval_start_utc", region_column, "fuel_type"]).reset_index(drop=True)


def _load_emission_factors(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["fuel_type", "emission_factor_tco2_per_mwh"])
    factors = pd.read_csv(path)
    factors["fuel_type"] = factors["fuel_type"].astype(str)
    factors["emission_factor_tco2_per_mwh"] = pd.to_numeric(
        factors["emission_factor_tco2_per_mwh"], errors="coerce"
    )
    return factors[["fuel_type", "emission_factor_tco2_per_mwh"]].dropna(subset=["fuel_type"])


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def build_fact_opportunity_displacement(
    fact_curtailment_opportunity_hourly: pd.DataFrame,
    fact_fossil_stack_hourly: pd.DataFrame,
    *,
    region_column: str = "parent_region",
    emission_factors: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    opportunity = fact_curtailment_opportunity_hourly.copy()
    if opportunity.empty:
        empty_hourly = pd.DataFrame()
        return {
            OPPORTUNITY_DISPLACEMENT_FUEL_HOURLY_TABLE: empty_hourly,
            OPPORTUNITY_DISPLACEMENT_HOURLY_TABLE: empty_hourly,
            OPPORTUNITY_DISPLACEMENT_DAILY_TABLE: empty_hourly,
        }

    opportunity["interval_start_utc"] = pd.to_datetime(opportunity["interval_start_utc"], utc=True, errors="coerce")
    opportunity["opportunity_deliverable_mwh"] = pd.to_numeric(
        opportunity.get("opportunity_deliverable_mwh"), errors="coerce"
    ).fillna(0.0)
    scoped = opportunity[
        opportunity["interval_start_utc"].notna()
        & opportunity[region_column].notna()
        & opportunity["opportunity_deliverable_mwh"].gt(0.0)
    ].copy()

    stack = fact_fossil_stack_hourly.copy()
    stack["interval_start_utc"] = pd.to_datetime(stack["interval_start_utc"], utc=True, errors="coerce")
    stack["fossil_generation_mwh"] = pd.to_numeric(stack.get("fossil_generation_mwh"), errors="coerce").fillna(0.0)
    stack["fossil_available_capacity_mwh_upper_bound"] = pd.to_numeric(
        stack.get("fossil_available_capacity_mwh_upper_bound"), errors="coerce"
    ).fillna(0.0)
    grouped_stack = {
        key: frame.copy()
        for key, frame in stack.groupby(["interval_start_utc", region_column], dropna=False)
    }

    fuel_factor = (
        emission_factors.copy()
        if emission_factors is not None and not emission_factors.empty
        else pd.DataFrame(columns=["fuel_type", "emission_factor_tco2_per_mwh"])
    )

    fuel_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    for row in scoped.itertuples(index=False):
        stack_key = (row.interval_start_utc, getattr(row, region_column))
        fossil_slice = grouped_stack.get(stack_key, pd.DataFrame())
        total_generation = float(_numeric_series(fossil_slice, "fossil_generation_mwh").sum())
        total_available = float(_numeric_series(fossil_slice, "fossil_available_capacity_mwh_upper_bound").sum())
        if total_generation > 0:
            allocation_basis = "physical_generation"
            allocation_pool = total_generation
            weights = _numeric_series(fossil_slice, "fossil_generation_mwh")
        elif total_available > 0:
            allocation_basis = "available_capacity"
            allocation_pool = total_available
            weights = _numeric_series(fossil_slice, "fossil_available_capacity_mwh_upper_bound")
        else:
            allocation_basis = "none"
            allocation_pool = 0.0
            weights = pd.Series(0.0, index=fossil_slice.index)

        displaced_total = min(float(row.opportunity_deliverable_mwh), allocation_pool)
        unresolved = max(float(row.opportunity_deliverable_mwh) - displaced_total, 0.0)

        hourly_rows.append(
            {
                "date": getattr(row, "date", pd.NA),
                "interval_start_utc": row.interval_start_utc,
                "cluster_key": getattr(row, "cluster_key", pd.NA),
                "cluster_label": getattr(row, "cluster_label", pd.NA),
                "parent_region": getattr(row, "parent_region", pd.NA),
                "route_name": getattr(row, "route_name", pd.NA),
                "route_label": getattr(row, "route_label", pd.NA),
                "route_border_key": getattr(row, "route_border_key", pd.NA),
                "route_target_zone": getattr(row, "route_target_zone", pd.NA),
                "hub_key": getattr(row, "hub_key", pd.NA),
                "hub_label": getattr(row, "hub_label", pd.NA),
                "opportunity_deliverable_mwh": float(row.opportunity_deliverable_mwh),
                "opportunity_gross_value_eur": pd.to_numeric(getattr(row, "opportunity_gross_value_eur", 0.0), errors="coerce"),
                "counterfactual_destination_kind": "route_target_zone_export_option",
                "route_loss_accounting_state": "embedded_in_route_score_heuristic",
                "fossil_displacement_region_scope": region_column,
                "fossil_displacement_region_key": getattr(row, region_column),
                "fossil_stack_generation_mwh": total_generation,
                "fossil_stack_available_capacity_mwh_upper_bound": total_available,
                "displacement_allocation_basis": allocation_basis,
                "displaced_fossil_mwh": displaced_total,
                "undisplaced_opportunity_mwh": unresolved,
                "source_lineage": "fact_curtailment_opportunity_hourly|fact_fossil_stack_hourly",
            }
        )

        if displaced_total <= 0 or fossil_slice.empty or weights.sum() <= 0:
            continue
        weights = weights / weights.sum()
        allocated = weights * displaced_total
        fuel_frame = fossil_slice[["fuel_type"]].copy()
        fuel_frame["allocated_displaced_fossil_mwh"] = allocated.to_numpy()
        if not fuel_factor.empty:
            fuel_frame = fuel_frame.merge(fuel_factor, on="fuel_type", how="left")
            fuel_frame["allocated_displaced_emissions_tco2"] = (
                fuel_frame["allocated_displaced_fossil_mwh"]
                * pd.to_numeric(fuel_frame["emission_factor_tco2_per_mwh"], errors="coerce")
            )
        else:
            fuel_frame["emission_factor_tco2_per_mwh"] = pd.NA
            fuel_frame["allocated_displaced_emissions_tco2"] = pd.NA
        for fuel_row in fuel_frame.itertuples(index=False):
            fuel_rows.append(
                {
                    "date": getattr(row, "date", pd.NA),
                    "interval_start_utc": row.interval_start_utc,
                    "cluster_key": getattr(row, "cluster_key", pd.NA),
                    "parent_region": getattr(row, "parent_region", pd.NA),
                    "route_name": getattr(row, "route_name", pd.NA),
                    "route_target_zone": getattr(row, "route_target_zone", pd.NA),
                    "hub_key": getattr(row, "hub_key", pd.NA),
                    "fuel_type": fuel_row.fuel_type,
                    "displacement_allocation_basis": allocation_basis,
                    "allocated_displaced_fossil_mwh": float(fuel_row.allocated_displaced_fossil_mwh),
                    "emission_factor_tco2_per_mwh": fuel_row.emission_factor_tco2_per_mwh,
                    "allocated_displaced_emissions_tco2": fuel_row.allocated_displaced_emissions_tco2,
                    "source_lineage": "fact_curtailment_opportunity_hourly|fact_fossil_stack_hourly",
                }
            )

    hourly = pd.DataFrame(hourly_rows)
    fuel_hourly = pd.DataFrame(fuel_rows)
    if not hourly.empty:
        hourly["opportunity_gross_value_eur"] = pd.to_numeric(hourly["opportunity_gross_value_eur"], errors="coerce")
    if fuel_hourly.empty:
        daily = pd.DataFrame()
    else:
        daily = (
            fuel_hourly.groupby(["date", "route_name", "route_target_zone", "hub_key", "fuel_type"], dropna=False)
            .agg(
                displaced_fossil_mwh=("allocated_displaced_fossil_mwh", "sum"),
                displaced_emissions_tco2=(
                    "allocated_displaced_emissions_tco2",
                    lambda values: pd.to_numeric(pd.Series(values), errors="coerce").sum(min_count=1),
                ),
            )
            .reset_index()
        )
        daily["source_lineage"] = "fact_opportunity_displacement_fuel_hourly"
    return {
        OPPORTUNITY_DISPLACEMENT_FUEL_HOURLY_TABLE: fuel_hourly,
        OPPORTUNITY_DISPLACEMENT_HOURLY_TABLE: hourly,
        OPPORTUNITY_DISPLACEMENT_DAILY_TABLE: daily,
    }


def _load_table_root(path: str | Path) -> Dict[str, pd.DataFrame]:
    root = Path(path)
    if root.is_file():
        raise ValueError("fleet input must be a directory")

    table_names = {
        "fact_bmu_generation_half_hourly": "fact_bmu_generation_half_hourly.csv",
        "fact_bmu_availability_half_hourly": "fact_bmu_availability_half_hourly.csv",
        "fact_bmu_bid_offer_half_hourly": "fact_bmu_bid_offer_half_hourly.csv",
        "fact_bmu_dispatch_acceptance_half_hourly": "fact_bmu_dispatch_acceptance_half_hourly.csv",
    }
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in table_names}

    if (root / table_names["fact_bmu_generation_half_hourly"]).exists():
        candidates = [root]
    else:
        candidates = [path for path in sorted(root.iterdir()) if path.is_dir()]

    for candidate in candidates:
        for table_name, filename in table_names.items():
            csv_path = candidate / filename
            if csv_path.exists():
                frames[table_name].append(pd.read_csv(csv_path))

    return {
        table_name: pd.concat(table_list, ignore_index=True) if table_list else pd.DataFrame()
        for table_name, table_list in frames.items()
    }


def materialize_reviewed_opportunity_displacement(
    opportunity_input_path: str | Path,
    fleet_input_path: str | Path,
    output_dir: str | Path,
    *,
    emission_factor_path: str | Path | None = None,
    region_column: str = "parent_region",
) -> Dict[str, pd.DataFrame]:
    opportunity = load_curtailment_opportunity_input(opportunity_input_path)
    fleet_tables = _load_table_root(fleet_input_path)
    fossil_stack = build_fact_fossil_stack_hourly(
        fleet_tables["fact_bmu_generation_half_hourly"],
        fact_bmu_availability_half_hourly=fleet_tables["fact_bmu_availability_half_hourly"],
        fact_bmu_bid_offer_half_hourly=fleet_tables["fact_bmu_bid_offer_half_hourly"],
        fact_bmu_dispatch_acceptance_half_hourly=fleet_tables["fact_bmu_dispatch_acceptance_half_hourly"],
        region_column=region_column,
    )
    frames = {
        FOSSIL_STACK_HOURLY_TABLE: fossil_stack,
        **build_fact_opportunity_displacement(
            opportunity,
            fossil_stack,
            region_column=region_column,
            emission_factors=_load_emission_factors(emission_factor_path),
        ),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(output_path / f"{table_name}.csv", index=False)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opportunity-input-path", required=True, help="Bundle directory or fact_curtailment_opportunity_hourly.csv path")
    parser.add_argument("--fleet-input-path", required=True, help="Single fleet-output dir or root containing monthly fleet subdirectories")
    parser.add_argument("--output-dir", required=True, help="Output directory for displacement tables")
    parser.add_argument(
        "--fuel-emission-factor-path",
        help="Optional CSV with columns fuel_type,emission_factor_tco2_per_mwh",
    )
    parser.add_argument(
        "--region-column",
        default="parent_region",
        choices=("parent_region",),
        help="Opportunity and BMU region column used for same-region displacement matching",
    )
    args = parser.parse_args()

    frames = materialize_reviewed_opportunity_displacement(
        opportunity_input_path=args.opportunity_input_path,
        fleet_input_path=args.fleet_input_path,
        output_dir=args.output_dir,
        emission_factor_path=args.fuel_emission_factor_path,
        region_column=args.region_column,
    )
    print(
        f"[source=reviewed_opportunity_displacement] Materialized {len(frames)} tables "
        f"opportunity={args.opportunity_input_path} fleet={args.fleet_input_path} output={args.output_dir}"
    )
    for table_name, frame in frames.items():
        print(f"{table_name}: rows={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
