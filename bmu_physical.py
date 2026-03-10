from __future__ import annotations

import datetime as dt
import urllib.parse
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from bmu_generation import (
    ElexonError,
    build_dim_bmu_asset,
    build_fact_bmu_generation_half_hourly,
    fetch_b1610_generation,
    fetch_bmu_reference_all,
)
from bmu_truth_utils import (
    ELEXON_BASE,
    build_bmu_interval_spine,
    fetch_json,
    local_date_range_to_utc_window,
    rfc3339_utc,
    unwrap_data_rows,
)


PHYSICAL_DATASETS = ("PN", "QPN", "MILS", "MELS")


def fetch_balancing_physical(
    elexon_bm_units: Sequence[str],
    start_date: dt.date,
    end_date: dt.date,
    datasets: Sequence[str] = PHYSICAL_DATASETS,
) -> pd.DataFrame:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not elexon_bm_units:
        raise ElexonError("no BMUs provided for balancing physical fetch")

    start_utc, end_utc = local_date_range_to_utc_window(start_date, end_date)
    frames = []
    for bm_unit in elexon_bm_units:
        params = [
            ("bmUnit", bm_unit),
            ("from", rfc3339_utc(start_utc)),
            ("to", rfc3339_utc(end_utc)),
        ]
        params.extend(("dataset", dataset) for dataset in datasets)
        url = f"{ELEXON_BASE}/balancing/physical?{urllib.parse.urlencode(params, doseq=True)}"
        rows = unwrap_data_rows(fetch_json(url))
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["requested_bm_unit"] = bm_unit
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pivot_dataset_columns(
    frame: pd.DataFrame,
    source_dataset: str,
    key_columns: list[str],
) -> pd.DataFrame:
    subset = frame[frame["source_dataset"] == source_dataset].copy()
    if subset.empty:
        return pd.DataFrame(columns=key_columns)

    subset = subset.sort_values(["time_from_utc", "time_to_utc"])
    subset = subset.groupby(key_columns, as_index=False, dropna=False).tail(1)
    rename_map = {
        "level_from_mw": f"{source_dataset.lower()}_level_from_mw",
        "level_to_mw": f"{source_dataset.lower()}_level_to_mw",
        "dataset_mean_mw": f"{source_dataset.lower()}_mean_mw",
        "dataset_energy_mwh": f"{source_dataset.lower()}_mwh",
        "time_from_utc": f"{source_dataset.lower()}_time_from_utc",
        "time_to_utc": f"{source_dataset.lower()}_time_to_utc",
    }
    return subset[key_columns + list(rename_map)].rename(columns=rename_map)


def build_fact_bmu_physical_position_half_hourly(
    dim_bmu_asset: pd.DataFrame,
    fact_bmu_generation_half_hourly: pd.DataFrame,
    raw_physical_frame: pd.DataFrame,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
    key_columns = ["settlement_date", "settlement_period", "elexon_bm_unit"]
    dataset_suffixes = ("level_from_mw", "level_to_mw", "mean_mw", "mwh", "time_from_utc", "time_to_utc")
    spine = build_bmu_interval_spine(dim_bmu_asset, start_date, end_date)
    if raw_physical_frame.empty:
        frame = spine.copy()
        for dataset in PHYSICAL_DATASETS:
            for suffix in dataset_suffixes:
                frame[f"{dataset.lower()}_{suffix}"] = pd.NA
    else:
        frame = raw_physical_frame.rename(
            columns={
                "dataset": "source_dataset",
                "settlementDate": "settlement_date",
                "settlementPeriod": "settlement_period",
                "timeFrom": "time_from_utc",
                "timeTo": "time_to_utc",
                "levelFrom": "level_from_mw",
                "levelTo": "level_to_mw",
                "bmUnit": "elexon_bm_unit",
                "nationalGridBmUnit": "national_grid_bm_unit_from_fact",
            }
        ).copy()
        frame["settlement_date"] = pd.to_datetime(frame["settlement_date"], errors="coerce").dt.date
        frame["settlement_period"] = pd.to_numeric(frame["settlement_period"], errors="coerce").astype("Int64")
        frame["time_from_utc"] = pd.to_datetime(frame["time_from_utc"], utc=True, errors="coerce")
        frame["time_to_utc"] = pd.to_datetime(frame["time_to_utc"], utc=True, errors="coerce")
        frame["level_from_mw"] = pd.to_numeric(frame["level_from_mw"], errors="coerce")
        frame["level_to_mw"] = pd.to_numeric(frame["level_to_mw"], errors="coerce")
        frame["interval_hours"] = (frame["time_to_utc"] - frame["time_from_utc"]).dt.total_seconds() / 3600.0
        frame["dataset_mean_mw"] = (frame["level_from_mw"] + frame["level_to_mw"]) / 2.0
        frame["dataset_energy_mwh"] = frame["dataset_mean_mw"] * frame["interval_hours"]
        frame = frame.dropna(subset=["settlement_date", "settlement_period", "elexon_bm_unit"])

        wide = spine.copy()
        for dataset in PHYSICAL_DATASETS:
            wide = wide.merge(_pivot_dataset_columns(frame, dataset, key_columns), on=key_columns, how="left")
        frame = wide

    for dataset in PHYSICAL_DATASETS:
        for suffix in dataset_suffixes:
            column_name = f"{dataset.lower()}_{suffix}"
            if column_name not in frame.columns:
                frame[column_name] = pd.NA

    generation = fact_bmu_generation_half_hourly[["settlement_date", "settlement_period", "elexon_bm_unit", "generation_mwh"]]
    frame = frame.merge(generation, on=key_columns, how="left")

    frame["source_key"] = "balancing_physical"
    frame["source_label"] = "Elexon balancing physical positions"
    frame["target_is_proxy"] = False
    frame["counterfactual_method"] = "pn_qpn_physical_max"

    frame["physical_baseline_mwh"] = frame[["pn_mwh", "qpn_mwh"]].max(axis=1, skipna=True)
    frame["physical_baseline_source_dataset"] = np.select(
        [
            frame["qpn_mwh"].notna() & ((frame["pn_mwh"].isna()) | (frame["qpn_mwh"] >= frame["pn_mwh"])),
            frame["pn_mwh"].notna(),
        ],
        ["QPN", "PN"],
        default=pd.NA,
    )
    frame["physical_consistency_flag"] = True
    frame.loc[
        frame["mils_mwh"].notna() & frame["physical_baseline_mwh"].notna() & (frame["physical_baseline_mwh"] + 1e-6 < frame["mils_mwh"]),
        "physical_consistency_flag",
    ] = False
    frame.loc[
        frame["mels_mwh"].notna() & frame["physical_baseline_mwh"].notna() & (frame["physical_baseline_mwh"] - 1e-6 > frame["mels_mwh"]),
        "physical_consistency_flag",
    ] = False

    frame["counterfactual_valid_flag"] = True
    frame.loc[frame["physical_baseline_mwh"].isna(), "counterfactual_valid_flag"] = False
    frame.loc[frame["generation_mwh"].isna(), "counterfactual_valid_flag"] = False
    frame.loc[
        frame["generation_mwh"].notna() & frame["physical_baseline_mwh"].notna() & (frame["physical_baseline_mwh"] + 1e-6 < frame["generation_mwh"]),
        "counterfactual_valid_flag",
    ] = False
    frame.loc[~frame["physical_consistency_flag"], "counterfactual_valid_flag"] = False

    keep_columns = [
        "settlement_date",
        "settlement_period",
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "source_key",
        "source_label",
        "target_is_proxy",
        "elexon_bm_unit",
        "national_grid_bm_unit",
        "bm_unit_name",
        "lead_party_name",
        "fuel_type",
        "bm_unit_type",
        "gsp_group_id",
        "gsp_group_name",
        "generation_capacity_mw",
        "mapping_status",
        "mapping_confidence",
        "mapping_rule",
        "cluster_key",
        "cluster_label",
        "parent_region",
        "generation_mwh",
        "pn_level_from_mw",
        "pn_level_to_mw",
        "pn_mean_mw",
        "pn_mwh",
        "qpn_level_from_mw",
        "qpn_level_to_mw",
        "qpn_mean_mw",
        "qpn_mwh",
        "mils_level_from_mw",
        "mils_level_to_mw",
        "mils_mean_mw",
        "mils_mwh",
        "mels_level_from_mw",
        "mels_level_to_mw",
        "mels_mean_mw",
        "mels_mwh",
        "physical_baseline_source_dataset",
        "physical_baseline_mwh",
        "physical_consistency_flag",
        "counterfactual_method",
        "counterfactual_valid_flag",
    ]
    frame = frame[keep_columns].sort_values(["interval_start_utc", "elexon_bm_unit"])
    frame = frame.drop_duplicates(subset=["settlement_date", "settlement_period", "elexon_bm_unit"], keep="last")
    return frame.reset_index(drop=True)


def materialize_bmu_physical_history(
    start_date: dt.date,
    end_date: dt.date,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    reference = fetch_bmu_reference_all()
    dim_bmu_asset = build_dim_bmu_asset(reference)
    raw_generation = fetch_b1610_generation(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    fact_bmu_generation_half_hourly = build_fact_bmu_generation_half_hourly(dim_bmu_asset, raw_generation)
    raw_physical = fetch_balancing_physical(dim_bmu_asset["elexon_bm_unit"].tolist(), start_date, end_date)
    fact_bmu_physical_position_half_hourly = build_fact_bmu_physical_position_half_hourly(
        dim_bmu_asset=dim_bmu_asset,
        fact_bmu_generation_half_hourly=fact_bmu_generation_half_hourly,
        raw_physical_frame=raw_physical,
        start_date=start_date,
        end_date=end_date,
    )

    frames = {
        "fact_bmu_physical_position_half_hourly": fact_bmu_physical_position_half_hourly,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in frames.items():
        frame.to_csv(target_dir / f"{table_name}.csv", index=False)
    return frames
