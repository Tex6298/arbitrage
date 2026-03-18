# Map Operational Posture

This document records the current operational decision for the curtailment map
against the authoritative local evidence set.

## Decision

Do **not** promote the map beyond exploratory use yet.

The earlier internal-operational posture based on the narrow local reviewed
stock is withdrawn after the broader supported-range sweep.

## Basis

The original narrow-stock decision was based on these authoritative outputs:

- `model_readiness_reviewed_bundle_batch_authoritative_rerun`
- `model_readiness_oct_2024_drift_policy_v3`
- `model_readiness_dec_2024_event_lifecycle`

Those outputs still show that the previously curated local windows are strong:

- all informative local reviewed windows are passing readiness
- October 1-7, 2024 is `7/7` ready days in the authoritative rerun
- December 7-9, 2024 is `3/3` ready days after event-lifecycle hardening
- November 22-24, 2024 is `3/3` ready days
- the retired `v3` specialist remains rejected in
  `model_readiness_gb_nl_shadow_suite_v1`

The broader supported-range sweep changes the decision:

- 19 reviewed windows were evaluated in `_local_runs/reviewed_volume_supported_range_v1`
- 13 of those windows were informative
- only 4 informative windows were fully ready
- 20 daily readiness failures remain in the broader sweep
- repeated failures cluster on `R2_netback_GB_NL_DE_PL`, especially
  `dogger_hornsea_offshore` and `moray_firth_offshore`, across multiple 2025
  windows

That is enough to reject a generalized internal-operational promotion.

## Operating Guardrails

Exploratory use should keep these guardrails in place:

- treat `fact_model_readiness_daily` as a hard gate
- only show map states as operational when `model_ready_flag = true`
- surface readiness state and blocking reasons in the UI or downstream consumer
- keep the map labeled as exploratory or experimental
- keep January 2025 out of promotion logic until better reviewed evidence exists

## What This Does Not Mean

This decision means:

- the narrow curated stock is still useful as a design proof
- the current baseline is not yet broad-range map-ready
- the remaining issue is broader regime coverage, not lack of tooling

## Next Engineering Program

The next program is now targeted regime work plus another broad batch rerun.

Priority order:

1. inspect the repeated 2025 `R2_netback_GB_NL_DE_PL` failures in the broader
   sweep
2. fix those regimes in the canonical `v2` path, not via a new specialist
3. rerun the same supported-range batch
4. only then reconsider internal-operational promotion
