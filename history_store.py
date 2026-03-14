from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


TRUTH_STORE_PRIMARY_KEYS: Dict[str, list[str]] = {
    "dim_interconnector_cable": ["connector_key"],
    "fact_interconnector_flow_hourly": ["interval_start_utc", "border_key", "direction_key"],
    "fact_interconnector_itl_hourly": ["interval_start_utc", "connector_key", "direction_key"],
    "fact_interconnector_capacity_hourly": [
        "interval_start_utc",
        "border_key",
        "direction_key",
        "auction_type",
        "contract_market_agreement_type",
    ],
    "fact_day_ahead_constraint_boundary_half_hourly": ["interval_start_utc", "boundary_key"],
    "fact_gb_transfer_boundary_reviewed_hourly": ["interval_start_utc", "cluster_key", "hub_key"],
    "fact_interconnector_capacity_reviewed_hourly": [
        "interval_start_utc",
        "border_key",
        "direction_key",
        "accepted_variant_key",
    ],
    "fact_interconnector_capacity_review_policy": ["border_key", "direction_key"],
    "fact_interconnector_capacity_source_audit_daily": ["border_key", "direction_key"],
    "fact_interconnector_capacity_source_audit_variant": ["border_key", "direction_key", "variant_key"],
    "fact_france_connector_operator_event": ["connector_key", "event_start_utc", "event_end_utc", "publish_time_utc"],
    "fact_france_connector_availability_hourly": ["interval_start_utc", "connector_key"],
    "fact_france_connector_operator_source_compare": [
        "requested_start_date",
        "requested_end_date",
        "connector_key",
        "source_variant_key",
    ],
    "fact_france_connector_reviewed_period": ["connector_key", "direction_key", "period_start_utc", "period_end_utc", "source_key"],
    "fact_france_connector_notice_hourly": ["interval_start_utc", "connector_key", "direction_key"],
    "fact_france_connector_hourly": ["interval_start_utc", "connector_key"],
    "fact_upstream_market_state_hourly": ["interval_start_utc", "route_name"],
    "fact_system_balance_market_state_hourly": ["interval_start_utc"],
    "fact_gb_transfer_gate_hourly": ["interval_start_utc", "cluster_key", "hub_key"],
    "fact_gb_transfer_reviewed_period": ["period_start_utc", "period_end_utc", "scope_key", "hub_key", "source_key"],
    "fact_gb_transfer_review_policy": ["source_key"],
    "fact_gb_transfer_reviewed_hourly": ["interval_start_utc", "cluster_key", "hub_key"],
    "fact_route_score_hourly": ["interval_start_utc", "cluster_key", "route_name", "hub_key"],
    "fact_curtailment_opportunity_hourly": ["interval_start_utc", "cluster_key", "route_name", "hub_key"],
    "fact_backtest_prediction_hourly": [
        "model_key",
        "forecast_horizon_hours",
        "interval_start_utc",
        "cluster_key",
        "route_name",
        "hub_key",
    ],
    "fact_backtest_summary_slice": ["model_key", "forecast_horizon_hours", "slice_dimension", "slice_value"],
    "fact_backtest_top_error_hourly": ["model_key", "forecast_horizon_hours", "top_error_rank"],
    "fact_drift_window": ["model_key", "forecast_horizon_hours", "window_start_utc", "drift_scope", "route_name", "cluster_key"],
    "fact_model_readiness_daily": ["window_date", "model_key"],
    "fact_bmu_bid_offer_half_hourly": ["settlement_date", "settlement_period", "elexon_bm_unit"],
    "fact_bmu_physical_position_half_hourly": ["settlement_date", "settlement_period", "elexon_bm_unit"],
    "fact_bmu_availability_half_hourly": ["settlement_date", "settlement_period", "elexon_bm_unit"],
    "fact_bmu_curtailment_truth_half_hourly": ["settlement_date", "settlement_period", "elexon_bm_unit"],
    "fact_curtailment_reconciliation_daily": ["settlement_date"],
    "fact_constraint_target_audit_daily": ["settlement_date"],
    "fact_dispatch_alignment_daily": ["settlement_date"],
    "fact_dispatch_alignment_bmu_daily": ["settlement_date", "elexon_bm_unit"],
    "fact_curtailment_gap_reason_daily": ["settlement_date", "lost_energy_block_reason"],
    "fact_bmu_curtailment_gap_bmu_daily": ["settlement_date", "elexon_bm_unit"],
    "fact_bmu_family_shortfall_daily": ["settlement_date", "bmu_family_key"],
    "fact_support_case_daily": ["support_batch_id", "settlement_date"],
    "fact_support_case_family_daily": ["support_batch_id", "settlement_date", "bmu_family_key"],
    "fact_support_case_half_hourly": ["support_batch_id", "settlement_date", "settlement_period", "elexon_bm_unit"],
    "fact_support_case_resolution": ["support_batch_id", "settlement_date", "bmu_family_key"],
    "fact_support_resolution_daily": ["support_batch_id", "settlement_date"],
    "fact_support_resolution_batch": ["support_batch_id"],
    "fact_support_rerun_gate_daily": ["support_batch_id", "settlement_date"],
    "fact_support_rerun_gate_batch": ["support_batch_id"],
    "fact_support_open_case_priority_family_daily": ["support_batch_id", "settlement_date", "bmu_family_key"],
    "fact_support_resolution_pattern_summary": ["support_batch_id", "support_resolution_pattern_key"],
    "fact_support_resolution_pattern_member_family_daily": ["support_batch_id", "settlement_date", "bmu_family_key"],
    "fact_support_rerun_candidate_daily": ["support_batch_id", "settlement_date"],
    "fact_support_rerun_candidate_family_daily": ["support_batch_id", "settlement_date", "bmu_family_key"],
}


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def _normalize_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    return value


def _prepare_frame(frame: pd.DataFrame, primary_keys: Iterable[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for key in primary_keys:
        if key not in prepared.columns:
            raise ValueError(f"missing primary key column '{key}'")
    prepared = prepared.drop_duplicates(subset=list(primary_keys), keep="last").reset_index(drop=True)
    return prepared


def _create_or_extend_table(
    connection: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
    primary_keys: list[str],
) -> None:
    existing_columns = {
        row[1]: row[2]
        for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    }
    if not existing_columns:
        column_defs = [
            f"{_quote_identifier(column)} {_sqlite_type(frame[column])}"
            for column in frame.columns
        ]
        primary_key_sql = ", ".join(_quote_identifier(column) for column in primary_keys)
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {_quote_identifier(table_name)} "
            f"({', '.join(column_defs)}, PRIMARY KEY ({primary_key_sql}))"
        )
        return

    for column in frame.columns:
        if column in existing_columns:
            continue
        connection.execute(
            f"ALTER TABLE {_quote_identifier(table_name)} "
            f"ADD COLUMN {_quote_identifier(column)} {_sqlite_type(frame[column])}"
        )


def upsert_frame_to_sqlite(
    db_path: str | Path,
    table_name: str,
    frame: pd.DataFrame,
    primary_keys: list[str],
) -> dict:
    prepared = _prepare_frame(frame, primary_keys)
    target_path = Path(db_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(target_path)) as connection:
        _create_or_extend_table(connection, table_name, prepared, primary_keys)
        if not prepared.empty:
            columns = list(prepared.columns)
            placeholders = ", ".join("?" for _ in columns)
            insert_sql = (
                f"INSERT OR REPLACE INTO {_quote_identifier(table_name)} "
                f"({', '.join(_quote_identifier(column) for column in columns)}) "
                f"VALUES ({placeholders})"
            )
            rows = [
                tuple(_normalize_scalar(value) for value in row)
                for row in prepared.itertuples(index=False, name=None)
            ]
            connection.executemany(insert_sql, rows)
            connection.commit()
        table_row_count = connection.execute(
            f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}"
        ).fetchone()[0]
    return {
        "table_name": table_name,
        "rows_loaded": len(prepared),
        "table_row_count": int(table_row_count),
    }


def upsert_truth_frames_to_sqlite(
    frames: Dict[str, pd.DataFrame],
    db_path: str | Path,
) -> pd.DataFrame:
    rows = []
    for table_name, primary_keys in TRUTH_STORE_PRIMARY_KEYS.items():
        if table_name not in frames:
            continue
        rows.append(upsert_frame_to_sqlite(db_path, table_name, frames[table_name], primary_keys))
    return pd.DataFrame(rows)


def ingest_truth_csv_tree_to_sqlite(
    root_dir: str | Path,
    db_path: str | Path,
) -> pd.DataFrame:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"CSV root directory does not exist: {root_path}")

    summary: Dict[str, dict] = {}
    for csv_path in sorted(root_path.rglob("*.csv")):
        table_name = csv_path.stem
        if table_name not in TRUTH_STORE_PRIMARY_KEYS:
            continue
        frame = pd.read_csv(csv_path)
        load_summary = upsert_frame_to_sqlite(
            db_path=db_path,
            table_name=table_name,
            frame=frame,
            primary_keys=TRUTH_STORE_PRIMARY_KEYS[table_name],
        )
        entry = summary.setdefault(
            table_name,
            {"table_name": table_name, "files_loaded": 0, "rows_loaded": 0, "table_row_count": 0},
        )
        entry["files_loaded"] += 1
        entry["rows_loaded"] += load_summary["rows_loaded"]
        entry["table_row_count"] = load_summary["table_row_count"]

    return pd.DataFrame(sorted(summary.values(), key=lambda row: row["table_name"]))
