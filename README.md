# Inline Arbitrage (Live) - GB via Elexon, Continent via ENTSO-E

This script fetches GB market-index prices from Elexon MID and continental
day-ahead prices from the ENTSO-E Transparency Platform for FR, NL, DE-LU, PL, CZ,
then computes simple route netbacks for:

- `GB->FR->DE->PL`
- `GB->NL->DE->PL`

Key behavior:

- Uses Elexon MID for GB and real ENTSO-E area EIC codes for the continental zones
- Queries by local market day for each zone, then converts timestamps to UTC
- Converts GB prices from GBP to EUR using a user-supplied FX rate
- Keeps physical-network assumptions in a separate module: `physical_constraints.py`
- Adds first-pass physical scaffolding in `asset_mapping.py` and `gb_topology.py`
- Adds a formal exploration plan in `exploration_plan.py` for historical data, map layers, backtests, and drift checks
- Adds `curtailment_signals.py` to materialize the first three historical tables for backtesting
- Adds `bmu_generation.py` to materialize BMU standing data and first-pass B1610 generation history
- Scores routes leg-by-leg and blocks a route when any border leg is underwater
- Uses synthetic data only when you ask for it with `--dry`

## Quick start

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then put your API credentials and FX rate in it:

   ```
   ENTOS_E_TOKEN=YOUR_TOKEN_HERE
   BMRS_API_KEY=YOUR_BMRS_API_KEY_HERE
   GBP_EUR=1.17
   ```

   `BMRS_API_KEY` is read exactly by that name if you have an Iris/Elexon key.
   Either `ENTOS_E_TOKEN` or `ENTSOE_TOKEN` works for the ENTSO-E token.
   Either `GBP_EUR` or `GBP_EUR_RATE` works for the FX rate.

3. Run a real pull for one market day:

   ```bash
   python inline_arbitrage_live.py --date 2025-09-20 --save out.csv
   ```

4. Or run synthetic demo data explicitly:

   ```bash
   python inline_arbitrage_live.py --date 2025-09-20 --dry
   ```

5. Print the current physical-network assumption register:

   ```bash
   python inline_arbitrage_live.py --show-constraint-assumptions
   ```

6. Print the current asset and topology scaffolding:

   ```bash
   python inline_arbitrage_live.py --show-asset-mapping
   python inline_arbitrage_live.py --show-gb-topology
   ```

7. Print the historical-data, map, and drift plan:

   ```bash
   python inline_arbitrage_live.py --show-exploration-plan
   ```

8. Materialize the first three historical tables:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-curtailment-history ^
     --history-year 2024-2025 ^
     --history-start 2024-10-01 ^
     --history-end 2024-10-07 ^
     --history-output-dir curtailment_history
   ```

9. Materialize BMU standing data and first-pass B1610 generation history:

   ```bash
   python inline_arbitrage_live.py ^
     --materialize-bmu-generation ^
     --bmu-start 2024-10-01 ^
     --bmu-end 2024-10-03 ^
     --bmu-output-dir bmu_history
   ```

## Notes

- GB prices come from the Elexon market-index feed and are published in GBP, so an FX rate
  is required to compare them against EUR-denominated continental markets.
- Continental ENTSO-E day-ahead prices are a local market-day product, not a generic rolling UTC feed.
- The GB feed is half-hourly and is normalized to hourly means before route calculation.
- The default GB provider is `APXMIDP`, which is the only provider returning non-zero live
  rows in the sample query I checked on March 10, 2026.
- If Elexon changes authentication behavior, the script will send `BMRS_API_KEY` when present.
- Some continental zones may publish sub-hourly prices. The script normalizes sub-hourly data to
  hourly means before calculating the simple route netbacks.
- Route scores are heuristics based on per-leg spreads, losses, and fees. They are
  stricter than a simple `PL - GB` spread because a negative intermediate leg blocks
  the route for that hour.
- `asset_mapping.py` is a seed registry only. It uses approximate user-provided anchors and
  capacities so the repo has a concrete place to hang cluster, weather, and curtailment work.
- `gb_topology.py` is also a coarse scaffold. It exposes cluster-to-hub reachability statuses,
  not a validated network model, PTDF model, or operational transfer limit.
- `exploration_plan.py` formalizes the next stage: historical datasets, interactive map layers,
  walk-forward backtests, and drift monitors. It is a plan surface, not an implemented UI.
- `curtailment_signals.py` now materializes:
  - `fact_constraint_daily`
  - `fact_wind_split_half_hourly`
  - `fact_regional_curtailment_hourly_proxy`
- The hourly regional table is explicitly a proxy. It allocates GB-wide daily curtailment by
  regional wind share, then distributes it intraday by wind shape and down to clusters by
  approximate capacity share.
- `bmu_generation.py` now materializes:
  - `dim_bmu_asset`
  - `fact_bmu_generation_half_hourly`
- The BMU dimension is a first pass. It maps known wind BMUs into the current cluster registry
  using explicit name rules and leaves everything else as `mapping_status=unmapped`.
- `fact_bmu_generation_half_hourly` is actual generation truth from Elexon B1610, not curtailment truth.
  The next missing layer is accepted dispatch-down or redispatch truth by BMU and settlement period.

## Next steps

- Replace the seed asset registry with confirmed wind farm, node, and owner metadata
- Turn the topology scaffold into actual transfer gates between clusters and hubs
- Build the historical curtailment pipeline and persist backtest outputs as first-class tables
- Start with a cluster-point time-slider map, then add hub arcs and error/drift layers
- Add physical flow and ATC checks
- Calibrate fee and capacity costs with auction history
- Add imbalance-risk premia and intraday updates
- Build the interactive dashboard only after the historical and drift surfaces exist
