# Reviewed Backfill and Displacement

This note records the additive work for two missing pieces:

- broader reviewed-window backfill beyond the sampled supported-range run
- first-pass opportunity-to-fossil displacement outputs

## Why this exists

The current broader evidence stock under `_local_runs/reviewed_volume_supported_range_v1`
is still a sampled supported-range run, not a continuous 2024-2026 backfill.

The current authoritative BMU truth snapshot also remains wind-focused. That is
good enough for curtailment truth and readiness, but not enough by itself to say
"this could have displaced fossil generation" without another data layer.

## New additive pieces

### Supported-range backfill

Use:

```powershell
python scripts/materialize_supported_range_monthly_backfill.py `
  --repo-root . `
  --run-root _local_runs/reviewed_full_range_monthly_v1 `
  --start-month 2024-10 `
  --end-month 2026-02 `
  --materialize-reviewed-bundles `
  --run-reviewed-bundle-batch-eval `
  --skip-existing
```

Optional flags:

- `--reviewed-window-days 7`
- `--materialize-bmu-truth`
- `--materialize-bmu-fleet`
- `--opportunity-truth-profile proxy|research|precision|all`
- `--plan-only`

This is intentionally additive. It does not replace the existing sampled
supported-range run.

Important source constraint:

- reviewed opportunity bundles are now tiled into contiguous windows of at most
  `7` days because the GB Elexon MID feed rejects larger `from/to` spans with
  HTTP `400`
- BMU truth and all-fuel fleet work can still run at the full-month level

Prerequisite:

- the reviewed opportunity path still needs `ENTOS_E_TOKEN` or `ENTSOE_TOKEN`
  plus the BMRS and FX envs already used by `inline_arbitrage_live.py`

### All-fuel BMU fleet materializer

Use:

```powershell
python bmu_fleet_history.py `
  --start 2024-10-01 `
  --end 2024-10-31 `
  --output-dir _local_runs/reviewed_full_range_monthly_v1/fleet_months/bmu_fleet_history_2024-10-01_2024-10-31
```

This widens the BMU standing-data and generation/dispatch/availability/bid path
from wind-only production BMUs to all production BMUs, using a conservative
parent-region fallback from GSP metadata when no curated cluster mapping exists.

This path uses public Elexon endpoints and does not depend on the ENTSOE/BMRS
envs used by the reviewed opportunity materializer.

It is additive and does not replace the wind-truth path in:

- `bmu_generation.py`
- `bmu_dispatch.py`
- `bmu_availability.py`
- `curtailment_truth.py`

### Opportunity displacement materializer

Use:

```powershell
python reviewed_opportunity_displacement.py `
  --opportunity-input-path curtailment_opportunity_live_britned_reviewed_rerun_2024-10-01_2024-10-07 `
  --fleet-input-path _local_runs/reviewed_full_range_monthly_v1/fleet_months `
  --output-dir _local_runs/reviewed_full_range_monthly_v1/displacement_oct_2024
```

Optional:

- `--fuel-emission-factor-path`

This writes:

- `fact_fossil_stack_hourly`
- `fact_opportunity_displacement_hourly`
- `fact_opportunity_displacement_fuel_hourly`
- `fact_opportunity_displacement_daily`

## Current limitations

- The route layer already embeds losses and fees heuristically in the route
  score. The displacement output does not add a second explicit line-loss model.
- The displacement join is currently **same-region GB fossil displacement**, not
  a foreign destination-unit stack. It keeps the export route and target-zone
  context, but the offset side is still a GB BMU same-region approximation.
- A real 2024-2026 displacement backfill still needs actual all-fuel BMU fleet
  outputs across that range.

## Working assumptions

- `parent_region` is the first safe matching scope for fossil displacement.
- Wind-truth and model-readiness remain the authoritative path for curtailment
  and opportunity.
- Fossil displacement is a second-stage analytic layer, not a readiness gate.
