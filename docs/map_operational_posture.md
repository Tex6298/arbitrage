# Map Operational Posture

This document records the current operational decision for the curtailment map
against the authoritative local evidence set.

## Decision

Promote the map from exploratory to **internal operational** use.

This is **not** an external or fully generalized production claim. It is an
internal operating posture backed by the current authoritative readiness and
batch outputs.

## Basis

The decision is based on these authoritative outputs:

- `model_readiness_reviewed_bundle_batch_authoritative_rerun`
- `model_readiness_oct_2024_drift_policy_v3`
- `model_readiness_dec_2024_event_lifecycle`

At this point:

- all informative local reviewed windows are passing readiness
- October 1-7, 2024 is `7/7` ready days in the authoritative rerun
- December 7-9, 2024 is `3/3` ready days after event-lifecycle hardening
- November 22-24, 2024 is `3/3` ready days
- the retired `v3` specialist remains rejected in
  `model_readiness_gb_nl_shadow_suite_v1`

The remaining weak area is January 24-26, 2025, but that window remains a
non-informative guardrail and data-coverage issue rather than evidence of a
current forecasting break in the active baseline.

## Operating Guardrails

Internal operational use should keep these guardrails in place:

- treat `fact_model_readiness_daily` as a hard gate
- only show map states as operational when `model_ready_flag = true`
- surface readiness state and blocking reasons in the UI or downstream consumer
- keep the map labeled as internal-only
- keep January 2025 out of promotion logic until better reviewed evidence exists

## What This Does Not Mean

This decision does not mean:

- the map is ready for external productization
- the repo has sufficient broad historical evidence to claim generalized
  performance across all later regimes
- Shetland reviewed dependency gaps are solved without new auditable source rows

## Next Engineering Program

The next program should be broader evidence acquisition and batch evaluation,
not more local tuning.

Priority order:

1. materialize additional later reviewed windows
2. scout them for informative signal
3. rerun the reviewed-bundle batch on the expanded stock
4. only after that, reconsider whether the internal operational map can be
   promoted further
