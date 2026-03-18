from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from asset_mapping import cluster_frame
from curtailment_opportunity import CURTAILMENT_OPPORTUNITY_TABLE
from model_readiness import MODEL_READINESS_TABLE
from opportunity_backtest import MODEL_POTENTIAL_RATIO_V2


EXPLORATORY_CLUSTER_MAP_POINT_TABLE = "dim_exploratory_cluster_map_point"
EXPLORATORY_CLUSTER_MAP_HOURLY_TABLE = "fact_exploratory_cluster_map_hourly"
EXPLORATORY_CLUSTER_MAP_HTML = "exploratory_cluster_map.html"


def _empty_exploratory_cluster_map_point_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cluster_key",
            "cluster_label",
            "parent_region",
            "approx_capacity_mw",
            "centroid_latitude",
            "centroid_longitude",
            "mapping_confidence",
            "preferred_hub_candidates",
            "connection_context",
            "curation_version",
        ]
    )


def _empty_exploratory_cluster_map_hourly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "interval_start_utc",
            "interval_end_utc",
            "window_date",
            "cluster_key",
            "cluster_label",
            "parent_region",
            "mapping_confidence",
            "centroid_latitude",
            "centroid_longitude",
            "preferred_hub_candidates",
            "connection_context",
            "curation_version",
            "route_count",
            "feasible_route_count",
            "export_candidate_route_count",
            "proxy_internal_route_count",
            "reviewed_internal_route_count",
            "positive_route_score_count",
            "active_hub_count",
            "active_hubs",
            "active_routes",
            "deliverable_mw_proxy_sum",
            "opportunity_deliverable_mwh_sum",
            "opportunity_gross_value_eur_sum",
            "top_route_price_score_eur_per_mwh",
            "model_key",
            "model_ready_flag",
            "model_readiness_state",
            "blocking_reasons",
        ]
    )


def build_dim_exploratory_cluster_map_point() -> pd.DataFrame:
    clusters = cluster_frame().copy()
    if clusters.empty:
        return _empty_exploratory_cluster_map_point_frame()
    return clusters[
        [
            "cluster_key",
            "cluster_label",
            "parent_region",
            "approx_capacity_mw",
            "centroid_latitude",
            "centroid_longitude",
            "mapping_confidence",
            "preferred_hub_candidates",
            "connection_context",
            "curation_version",
        ]
    ].copy()


def _select_map_readiness_model(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    readiness = frame.copy()
    readiness["model_key"] = readiness["model_key"].fillna("").astype(str)
    preferred = readiness["model_key"].eq(MODEL_POTENTIAL_RATIO_V2)
    if preferred.any():
        readiness = readiness[preferred].copy()
    else:
        selected_model_key = readiness["model_key"].dropna().astype(str).sort_values().iloc[0]
        readiness = readiness[readiness["model_key"].eq(selected_model_key)].copy()
    return readiness


def build_fact_exploratory_cluster_map_hourly(
    fact_curtailment_opportunity_hourly: pd.DataFrame,
    fact_model_readiness_daily: pd.DataFrame,
) -> pd.DataFrame:
    if fact_curtailment_opportunity_hourly is None or fact_curtailment_opportunity_hourly.empty:
        return _empty_exploratory_cluster_map_hourly_frame()

    opportunity = fact_curtailment_opportunity_hourly.copy()
    opportunity["interval_start_utc"] = pd.to_datetime(opportunity["interval_start_utc"], utc=True, errors="coerce")
    opportunity["interval_end_utc"] = pd.to_datetime(opportunity["interval_end_utc"], utc=True, errors="coerce")
    opportunity = opportunity.dropna(subset=["interval_start_utc", "cluster_key"]).copy()
    if opportunity.empty:
        return _empty_exploratory_cluster_map_hourly_frame()

    opportunity["window_date"] = opportunity["interval_start_utc"].dt.floor("D")
    for column, default in (
        ("route_price_feasible_flag", False),
        ("export_candidate_flag", False),
        ("route_price_score_eur_per_mwh", 0.0),
        ("deliverable_mw_proxy", 0.0),
        ("opportunity_deliverable_mwh", 0.0),
        ("opportunity_gross_value_eur", 0.0),
        ("route_name", ""),
        ("hub_key", ""),
        ("internal_transfer_evidence_tier", "gb_topology_transfer_gate_proxy"),
    ):
        if column not in opportunity.columns:
            opportunity[column] = default

    opportunity["route_price_feasible_flag"] = opportunity["route_price_feasible_flag"].fillna(False).astype(bool)
    opportunity["export_candidate_flag"] = opportunity["export_candidate_flag"].fillna(False).astype(bool)
    opportunity["route_price_score_eur_per_mwh"] = pd.to_numeric(
        opportunity["route_price_score_eur_per_mwh"],
        errors="coerce",
    ).fillna(0.0)
    opportunity["deliverable_mw_proxy"] = pd.to_numeric(opportunity["deliverable_mw_proxy"], errors="coerce").fillna(0.0)
    opportunity["opportunity_deliverable_mwh"] = pd.to_numeric(
        opportunity["opportunity_deliverable_mwh"],
        errors="coerce",
    ).fillna(0.0)
    opportunity["opportunity_gross_value_eur"] = pd.to_numeric(
        opportunity["opportunity_gross_value_eur"],
        errors="coerce",
    ).fillna(0.0)

    points = build_dim_exploratory_cluster_map_point()
    opportunity = opportunity.merge(
        points,
        on=["cluster_key", "cluster_label", "parent_region"],
        how="left",
        suffixes=("", "_point"),
    )
    if opportunity.empty:
        return _empty_exploratory_cluster_map_hourly_frame()

    opportunity["proxy_internal_route_flag"] = opportunity["internal_transfer_evidence_tier"].fillna("").eq(
        "gb_topology_transfer_gate_proxy"
    )
    opportunity["reviewed_internal_route_flag"] = ~opportunity["proxy_internal_route_flag"]
    opportunity["positive_route_score_flag"] = opportunity["route_price_score_eur_per_mwh"].gt(0)

    grouped = (
        opportunity.groupby(
            [
                "interval_start_utc",
                "interval_end_utc",
                "window_date",
                "cluster_key",
                "cluster_label",
                "parent_region",
                "mapping_confidence",
                "centroid_latitude",
                "centroid_longitude",
                "preferred_hub_candidates",
                "connection_context",
                "curation_version",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            route_count=("cluster_key", "size"),
            feasible_route_count=("route_price_feasible_flag", "sum"),
            export_candidate_route_count=("export_candidate_flag", "sum"),
            proxy_internal_route_count=("proxy_internal_route_flag", "sum"),
            reviewed_internal_route_count=("reviewed_internal_route_flag", "sum"),
            positive_route_score_count=("positive_route_score_flag", "sum"),
            active_hub_count=("hub_key", lambda values: int(pd.Series(values)[pd.Series(values).astype(str).ne("")].nunique())),
            active_hubs=(
                "hub_key",
                lambda values: ", ".join(
                    sorted({str(value) for value in values if pd.notna(value) and str(value).strip()})
                ),
            ),
            active_routes=(
                "route_name",
                lambda values: ", ".join(
                    sorted({str(value) for value in values if pd.notna(value) and str(value).strip()})
                ),
            ),
            deliverable_mw_proxy_sum=("deliverable_mw_proxy", "sum"),
            opportunity_deliverable_mwh_sum=("opportunity_deliverable_mwh", "sum"),
            opportunity_gross_value_eur_sum=("opportunity_gross_value_eur", "sum"),
            top_route_price_score_eur_per_mwh=("route_price_score_eur_per_mwh", "max"),
        )
    )

    readiness = fact_model_readiness_daily.copy() if fact_model_readiness_daily is not None else pd.DataFrame()
    if readiness.empty:
        grouped["model_key"] = MODEL_POTENTIAL_RATIO_V2
        grouped["model_ready_flag"] = False
        grouped["model_readiness_state"] = "readiness_unavailable"
        grouped["blocking_reasons"] = "readiness_input_missing"
        return grouped[_empty_exploratory_cluster_map_hourly_frame().columns].copy()

    readiness["window_date"] = pd.to_datetime(readiness["window_date"], utc=True, errors="coerce").dt.floor("D")
    readiness = readiness.dropna(subset=["window_date"]).copy()
    readiness = _select_map_readiness_model(readiness)
    readiness = readiness[
        [
            "window_date",
            "model_key",
            "model_ready_flag",
            "model_readiness_state",
            "blocking_reasons",
        ]
    ].drop_duplicates(["window_date"], keep="last")
    readiness["model_ready_flag"] = readiness["model_ready_flag"].fillna(False).astype(bool)
    readiness["model_readiness_state"] = readiness["model_readiness_state"].fillna("readiness_unknown")
    readiness["blocking_reasons"] = readiness["blocking_reasons"].fillna("")

    grouped = grouped.merge(readiness, on="window_date", how="left")
    grouped["model_key"] = grouped["model_key"].fillna(MODEL_POTENTIAL_RATIO_V2)
    grouped["model_ready_flag"] = grouped["model_ready_flag"].fillna(False).astype(bool)
    grouped["model_readiness_state"] = grouped["model_readiness_state"].fillna("readiness_unknown")
    grouped["blocking_reasons"] = grouped["blocking_reasons"].fillna("")

    return grouped[_empty_exploratory_cluster_map_hourly_frame().columns].sort_values(
        ["interval_start_utc", "cluster_key"]
    ).reset_index(drop=True)


def _table_path(input_path: str | Path, table_name: str) -> Path:
    path = Path(input_path)
    if path.is_file():
        return path
    return path / f"{table_name}.csv"


def _load_table(input_path: str | Path, table_name: str) -> pd.DataFrame:
    csv_path = _table_path(input_path, table_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {table_name}.csv under {input_path}")
    return pd.read_csv(csv_path)


def _frame_json(frame: pd.DataFrame) -> str:
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = pd.to_datetime(serializable[column], utc=True, errors="coerce")
    return serializable.to_json(orient="records", date_format="iso")


def render_exploratory_cluster_map_html(
    dim_exploratory_cluster_map_point: pd.DataFrame,
    fact_exploratory_cluster_map_hourly: pd.DataFrame,
    output_path: str | Path,
) -> None:
    points_json = _frame_json(dim_exploratory_cluster_map_point)
    hourly_json = _frame_json(fact_exploratory_cluster_map_hourly)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Exploratory Cluster Map</title>
  <style>
    :root {{
      --ink: #19242c;
      --muted: #6f7c82;
      --paper: #f4efe3;
      --panel: rgba(255, 252, 246, 0.94);
      --line: rgba(25, 36, 44, 0.12);
      --sea-a: #dbe7ea;
      --sea-b: #b8ced3;
      --land-a: #f5e7bf;
      --land-b: #e2c88b;
      --scotland: #1e6b7a;
      --england: #b65c2d;
      --ready: #2e7d32;
      --not-ready: #9f3b22;
      --unknown: #6f7c82;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Trebuchet MS", "Gill Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.75), transparent 35%),
        linear-gradient(180deg, #efe8d6 0%, #e6dcc8 48%, #d9cfbb 100%);
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 2.1fr) minmax(320px, 1fr);
      gap: 20px;
      padding: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 16px 40px rgba(25, 36, 44, 0.08);
      backdrop-filter: blur(10px);
    }}
    .map-panel {{
      overflow: hidden;
      display: grid;
      grid-template-rows: auto auto minmax(420px, 1fr);
    }}
    .header {{
      padding: 20px 22px 8px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-family: Georgia, "Palatino Linotype", serif;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.05;
      font-weight: 600;
    }}
    .subhead {{
      margin-top: 10px;
      color: var(--muted);
      max-width: 58ch;
    }}
    .status-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 0 22px 12px;
    }}
    .controls {{
      display: grid;
      gap: 8px;
      padding: 0 22px 16px;
    }}
    .control-row {{
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    .time-label {{
      font-size: 18px;
      font-weight: 600;
    }}
    .range {{
      width: 100%;
      accent-color: #215b68;
    }}
    .map-stage {{
      position: relative;
      margin: 0 14px 14px;
      border-radius: 18px;
      overflow: hidden;
      background:
        linear-gradient(180deg, var(--sea-a) 0%, var(--sea-b) 100%);
      border: 1px solid rgba(25, 36, 44, 0.08);
    }}
    .map-stage::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        radial-gradient(circle at 34% 52%, rgba(245, 231, 191, 0.86), rgba(245, 231, 191, 0) 28%),
        radial-gradient(circle at 60% 38%, rgba(226, 200, 139, 0.58), rgba(226, 200, 139, 0) 18%),
        linear-gradient(90deg, transparent 0 9.5%, rgba(255,255,255,0.15) 9.5% 10%, transparent 10% 19.5%, rgba(255,255,255,0.15) 19.5% 20%, transparent 20% 29.5%, rgba(255,255,255,0.15) 29.5% 30%, transparent 30% 39.5%, rgba(255,255,255,0.15) 39.5% 40%, transparent 40% 49.5%, rgba(255,255,255,0.15) 49.5% 50%, transparent 50% 59.5%, rgba(255,255,255,0.15) 59.5% 60%, transparent 60% 69.5%, rgba(255,255,255,0.15) 69.5% 70%, transparent 70% 79.5%, rgba(255,255,255,0.15) 79.5% 80%, transparent 80% 89.5%, rgba(255,255,255,0.15) 89.5% 90%, transparent 90% 100%),
        linear-gradient(180deg, transparent 0 16.3%, rgba(255,255,255,0.15) 16.3% 16.6%, transparent 16.6% 33%, rgba(255,255,255,0.15) 33% 33.3%, transparent 33.3% 49.6%, rgba(255,255,255,0.15) 49.6% 49.9%, transparent 49.9% 66.3%, rgba(255,255,255,0.15) 66.3% 66.6%, transparent 66.6% 83%, rgba(255,255,255,0.15) 83% 83.3%, transparent 83.3% 100%);
      opacity: 0.65;
      pointer-events: none;
    }}
    .marker-layer {{
      position: relative;
      min-height: 520px;
    }}
    .marker {{
      position: absolute;
      transform: translate(-50%, -50%);
      border: 0;
      background: transparent;
      cursor: pointer;
      padding: 0;
      text-align: center;
      color: var(--ink);
    }}
    .marker-dot {{
      border-radius: 999px;
      border: 3px solid rgba(255,255,255,0.92);
      box-shadow: 0 8px 24px rgba(25, 36, 44, 0.18);
      transition: transform 140ms ease, box-shadow 140ms ease;
      margin: 0 auto 6px;
    }}
    .marker:hover .marker-dot,
    .marker.active .marker-dot {{
      transform: scale(1.08);
      box-shadow: 0 12px 28px rgba(25, 36, 44, 0.24);
    }}
    .marker-label {{
      display: inline-block;
      max-width: 180px;
      font-size: 12px;
      line-height: 1.2;
      font-weight: 600;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(255, 252, 246, 0.84);
      border: 1px solid rgba(25, 36, 44, 0.1);
    }}
    .sidebar {{
      padding: 18px;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .card {{
      padding: 16px 16px 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.64);
      border: 1px solid var(--line);
    }}
    .card h2,
    .card h3 {{
      margin: 0 0 8px;
      font-family: Georgia, "Palatino Linotype", serif;
      font-weight: 600;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.02em;
      border: 1px solid transparent;
      text-transform: capitalize;
    }}
    .badge.confidence-high {{ background: rgba(46, 125, 50, 0.12); color: #245c28; border-color: rgba(46, 125, 50, 0.2); }}
    .badge.confidence-medium {{ background: rgba(30, 107, 122, 0.12); color: #174f59; border-color: rgba(30, 107, 122, 0.2); }}
    .badge.confidence-low {{ background: rgba(182, 92, 45, 0.12); color: #8a431f; border-color: rgba(182, 92, 45, 0.2); }}
    .badge.readiness-ready_for_map {{ background: rgba(46, 125, 50, 0.14); color: var(--ready); border-color: rgba(46, 125, 50, 0.22); }}
    .badge.readiness-not_ready {{ background: rgba(159, 59, 34, 0.14); color: var(--not-ready); border-color: rgba(159, 59, 34, 0.22); }}
    .badge.readiness-readiness_unknown,
    .badge.readiness-readiness_unavailable {{ background: rgba(111, 124, 130, 0.14); color: var(--unknown); border-color: rgba(111, 124, 130, 0.2); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      padding: 12px;
      border-radius: 14px;
      background: rgba(244, 239, 227, 0.9);
      border: 1px solid rgba(25, 36, 44, 0.08);
    }}
    .stat-label {{
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .meta {{
      display: grid;
      gap: 8px;
      font-size: 14px;
    }}
    .meta strong {{
      display: inline-block;
      min-width: 110px;
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 1100px) {{
      .app {{
        grid-template-columns: 1fr;
      }}
      .marker-layer {{
        min-height: 440px;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <section class="panel map-panel">
      <div class="header">
        <div class="eyebrow">Exploratory Only</div>
        <h1>GB Cluster Time Slider</h1>
        <div class="subhead">
          This is a cluster-point exploration surface, not an operational map. It shows current opportunity intensity on the seed spatial scaffold and badges each cluster with mapping confidence plus the daily readiness gate.
        </div>
      </div>
      <div class="status-bar" id="status-bar"></div>
      <div class="controls">
        <div class="control-row">
          <div class="time-label" id="time-label">No interval selected</div>
          <div class="hint" id="summary-label">Waiting for map data.</div>
        </div>
        <input id="time-range" class="range" type="range" min="0" max="0" value="0">
      </div>
      <div class="map-stage">
        <div class="marker-layer" id="marker-layer"></div>
      </div>
    </section>
    <aside class="panel sidebar">
      <section class="card">
        <h2 id="cluster-title">Cluster detail</h2>
        <div class="badge-row" id="cluster-badges"></div>
        <div class="stats">
          <div class="stat">
            <div class="stat-label">Deliverable</div>
            <div class="stat-value" id="stat-deliverable">0.0</div>
          </div>
          <div class="stat">
            <div class="stat-label">Gross Value</div>
            <div class="stat-value" id="stat-gross-value">0.0</div>
          </div>
          <div class="stat">
            <div class="stat-label">Feasible Routes</div>
            <div class="stat-value" id="stat-feasible-routes">0</div>
          </div>
          <div class="stat">
            <div class="stat-label">Reviewed Routes</div>
            <div class="stat-value" id="stat-reviewed-routes">0</div>
          </div>
        </div>
      </section>
      <section class="card">
        <h3>Cluster Context</h3>
        <div class="meta">
          <div><strong>Region</strong><span id="meta-region"></span></div>
          <div><strong>Preferred hubs</strong><span id="meta-hubs"></span></div>
          <div><strong>Active hubs</strong><span id="meta-active-hubs"></span></div>
          <div><strong>Routes</strong><span id="meta-active-routes"></span></div>
          <div><strong>Connection</strong><span id="meta-connection"></span></div>
        </div>
      </section>
      <section class="card">
        <h3>Daily Readiness</h3>
        <div class="hint" id="readiness-summary"></div>
      </section>
    </aside>
  </div>
  <script>
    const pointRows = {points_json};
    const hourlyRows = {hourly_json};
    const markerLayer = document.getElementById("marker-layer");
    const timeRange = document.getElementById("time-range");
    const timeLabel = document.getElementById("time-label");
    const summaryLabel = document.getElementById("summary-label");
    const statusBar = document.getElementById("status-bar");
    const clusterTitle = document.getElementById("cluster-title");
    const clusterBadges = document.getElementById("cluster-badges");
    const readinessSummary = document.getElementById("readiness-summary");

    const timeKeys = [...new Set(hourlyRows.map((row) => row.interval_start_utc))].sort();
    const groupedByTime = new Map();
    for (const row of hourlyRows) {{
      if (!groupedByTime.has(row.interval_start_utc)) {{
        groupedByTime.set(row.interval_start_utc, new Map());
      }}
      groupedByTime.get(row.interval_start_utc).set(row.cluster_key, row);
    }}

    const bounds = pointRows.reduce((acc, row) => {{
      const lat = Number(row.centroid_latitude);
      const lon = Number(row.centroid_longitude);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {{
        return acc;
      }}
      return {{
        minLat: Math.min(acc.minLat, lat),
        maxLat: Math.max(acc.maxLat, lat),
        minLon: Math.min(acc.minLon, lon),
        maxLon: Math.max(acc.maxLon, lon),
      }};
    }}, {{ minLat: Infinity, maxLat: -Infinity, minLon: Infinity, maxLon: -Infinity }});

    let selectedClusterKey = pointRows.length ? pointRows[0].cluster_key : "";

    function confidenceClass(value) {{
      return `confidence-${{String(value || "unknown").toLowerCase()}}`;
    }}

    function readinessClass(value) {{
      return `readiness-${{String(value || "readiness_unknown").toLowerCase()}}`;
    }}

    function badge(label, value, cssClass) {{
      return `<span class="badge ${{cssClass}}">${{label}}: ${{value}}</span>`;
    }}

    function projectPoint(row) {{
      const lon = Number(row.centroid_longitude);
      const lat = Number(row.centroid_latitude);
      const width = markerLayer.clientWidth || 1000;
      const height = markerLayer.clientHeight || 520;
      const x = ((lon - bounds.minLon) / Math.max(bounds.maxLon - bounds.minLon, 0.0001)) * (width * 0.82) + width * 0.09;
      const y = (1 - ((lat - bounds.minLat) / Math.max(bounds.maxLat - bounds.minLat, 0.0001))) * (height * 0.78) + height * 0.12;
      return {{ x, y }};
    }}

    function markerColor(point, hourly) {{
      const deliverable = Number(hourly?.opportunity_deliverable_mwh_sum || 0);
      if (deliverable <= 0) {{
        return point.parent_region === "Scotland" ? "rgba(30, 107, 122, 0.55)" : "rgba(182, 92, 45, 0.55)";
      }}
      if (deliverable >= 100) {{
        return point.parent_region === "Scotland" ? "#0f5462" : "#8f461f";
      }}
      return point.parent_region === "Scotland" ? "#2a8492" : "#ca6d38";
    }}

    function markerSize(hourly) {{
      const deliverable = Number(hourly?.opportunity_deliverable_mwh_sum || 0);
      return Math.max(18, Math.min(56, 18 + Math.sqrt(deliverable) * 2.2));
    }}

    function currentRows() {{
      const key = timeKeys[Number(timeRange.value) || 0];
      return groupedByTime.get(key) || new Map();
    }}

    function currentPointRow(clusterKey) {{
      return pointRows.find((row) => row.cluster_key === clusterKey);
    }}

    function renderMarkers() {{
      const rows = currentRows();
      markerLayer.innerHTML = "";
      for (const point of pointRows) {{
        const hourly = rows.get(point.cluster_key) || {{ cluster_key: point.cluster_key }};
        const pos = projectPoint(point);
        const size = markerSize(hourly);
        const button = document.createElement("button");
        button.className = "marker" + (point.cluster_key === selectedClusterKey ? " active" : "");
        button.style.left = `${{pos.x}}px`;
        button.style.top = `${{pos.y}}px`;
        button.innerHTML = `
          <div class="marker-dot" style="width:${{size}}px;height:${{size}}px;background:${{markerColor(point, hourly)}};"></div>
          <div class="marker-label">${{point.cluster_label}}</div>
        `;
        button.addEventListener("click", () => {{
          selectedClusterKey = point.cluster_key;
          render();
        }});
        markerLayer.appendChild(button);
      }}
    }}

    function renderHeader() {{
      if (!timeKeys.length) {{
        timeLabel.textContent = "No interval selected";
        summaryLabel.textContent = "No map data available.";
        statusBar.innerHTML = badge("Mode", "exploratory only", "readiness-readiness_unknown");
        return;
      }}
      const rows = [...currentRows().values()];
      const first = rows[0] || null;
      const deliverable = rows.reduce((sum, row) => sum + Number(row.opportunity_deliverable_mwh_sum || 0), 0);
      const positiveClusters = rows.filter((row) => Number(row.opportunity_deliverable_mwh_sum || 0) > 0).length;
      const intervalKey = timeKeys[Number(timeRange.value) || 0];
      timeLabel.textContent = new Date(intervalKey).toLocaleString("en-GB", {{
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "UTC",
      }}) + " UTC";
      summaryLabel.textContent = `${{positiveClusters}} active clusters, ${{deliverable.toFixed(1)}} MWh deliverable across the scaffold.`;
      statusBar.innerHTML = [
        badge("Daily gate", first?.model_readiness_state || "readiness_unknown", readinessClass(first?.model_readiness_state || "readiness_unknown")),
        badge("Model", first?.model_key || "{MODEL_POTENTIAL_RATIO_V2}", "readiness-readiness_unknown"),
        badge("Mode", "exploratory only", "readiness-readiness_unknown"),
      ].join("");
    }}

    function renderSidebar() {{
      const point = currentPointRow(selectedClusterKey);
      const hourly = currentRows().get(selectedClusterKey) || null;
      if (!point) {{
        clusterTitle.textContent = "Cluster detail";
        clusterBadges.innerHTML = "";
        readinessSummary.textContent = "No cluster selected.";
        return;
      }}
      clusterTitle.textContent = point.cluster_label;
      clusterBadges.innerHTML = [
        badge("Confidence", point.mapping_confidence || "unknown", confidenceClass(point.mapping_confidence || "unknown")),
        badge("Readiness", hourly?.model_readiness_state || "readiness_unknown", readinessClass(hourly?.model_readiness_state || "readiness_unknown")),
      ].join("");
      document.getElementById("stat-deliverable").textContent = Number(hourly?.opportunity_deliverable_mwh_sum || 0).toFixed(1);
      document.getElementById("stat-gross-value").textContent = Number(hourly?.opportunity_gross_value_eur_sum || 0).toFixed(0);
      document.getElementById("stat-feasible-routes").textContent = String(hourly?.feasible_route_count || 0);
      document.getElementById("stat-reviewed-routes").textContent = String(hourly?.reviewed_internal_route_count || 0);
      document.getElementById("meta-region").textContent = point.parent_region || "";
      document.getElementById("meta-hubs").textContent = point.preferred_hub_candidates || "";
      document.getElementById("meta-active-hubs").textContent = hourly?.active_hubs || "none";
      document.getElementById("meta-active-routes").textContent = hourly?.active_routes || "none";
      document.getElementById("meta-connection").textContent = point.connection_context || "";
      const blockers = hourly?.blocking_reasons ? `Blocking reasons: ${{hourly.blocking_reasons}}.` : "No blocking reasons recorded for this day.";
      readinessSummary.textContent = `${{hourly?.model_key || "{MODEL_POTENTIAL_RATIO_V2}"}} on ${{(hourly?.window_date || "").slice(0, 10)}} is ${{hourly?.model_readiness_state || "readiness_unknown"}}. ${{blockers}}`;
    }}

    function render() {{
      renderHeader();
      renderMarkers();
      renderSidebar();
    }}

    if (timeKeys.length) {{
      timeRange.max = String(timeKeys.length - 1);
      timeRange.value = "0";
      timeRange.addEventListener("input", render);
      window.addEventListener("resize", render);
    }}
    render();
  </script>
</body>
</html>
"""
    Path(output_path).write_text(html, encoding="utf-8")


def materialize_exploratory_cluster_map(
    opportunity_input_path: str | Path,
    readiness_input_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, pd.DataFrame]:
    opportunity = _load_table(opportunity_input_path, CURTAILMENT_OPPORTUNITY_TABLE)
    readiness = _load_table(readiness_input_path, MODEL_READINESS_TABLE)
    points = build_dim_exploratory_cluster_map_point()
    hourly = build_fact_exploratory_cluster_map_hourly(
        fact_curtailment_opportunity_hourly=opportunity,
        fact_model_readiness_daily=readiness,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    points.to_csv(output_path / f"{EXPLORATORY_CLUSTER_MAP_POINT_TABLE}.csv", index=False)
    hourly.to_csv(output_path / f"{EXPLORATORY_CLUSTER_MAP_HOURLY_TABLE}.csv", index=False)
    render_exploratory_cluster_map_html(
        dim_exploratory_cluster_map_point=points,
        fact_exploratory_cluster_map_hourly=hourly,
        output_path=output_path / EXPLORATORY_CLUSTER_MAP_HTML,
    )
    return {
        EXPLORATORY_CLUSTER_MAP_POINT_TABLE: points,
        EXPLORATORY_CLUSTER_MAP_HOURLY_TABLE: hourly,
    }
