# MOCAP Cost Model — Frozen Units & Pricing Assumptions

Status: FROZEN for the current evaluation cycle. Any change to the values
below invalidates prior calibration models and experiment logs — retrain
and re-run experiments if these change.

## Resource units

| Resource | Unit | Where it's measured |
|---|---|---|
| CPU | aggregate CPU-seconds (not multiplied by core count again) | `mocap/cost/estimator.py` approximates as `wall_clock_seconds × num_cores`; real per-executor CPU time is not yet available (see Limitations) |
| I/O | bytes scanned from storage | Spark SQL metric `"size of files read"`, summed over the executed physical plan (`mocap/cost/estimator.py::_walk_sql_metrics`) |
| Shuffle | bytes moved through the shuffle (read + write) | Spark SQL metrics `"shuffle bytes written"`, `"remote bytes read"`, `"local bytes read"`, summed |

## Pricing rates (`config/pricing.json`)

| Rate | Value | Meaning |
|---|---|---|
| `cost_per_vcore_hour` | 0.048 USD | CPU cost = (CPU-seconds / 3600) × this rate |
| `cost_per_gb_scanned` | 0.005 USD / GiB | I/O cost = bytes_scanned in GiB × this rate |
| `cost_per_gb_shuffled` | 0.010 USD / GiB | Shuffle cost = bytes_shuffled in GiB × this rate |
| `cost_per_gb_hour_memory` | 0.006 USD | reserved, not currently used in `total_cost()` |
| `penalty_per_second` | 0.0 | reserved, not currently used |

`total_cost = cpu_cost + io_cost + shuffle_cost`. These are illustrative
cloud-analytics rates, not billing from a real provider account — the
paper must not describe them as real cloud billing (per the project's
scientific reporting rules).

## Pre-execution prediction vs. post-execution telemetry

Two distinct pipelines produce a cost/latency number, and the paper must
not conflate them:

- **`mocap/cost/predictor.py`** — pre-execution. Converts structural plan
  features (join/exchange/sort/aggregate counts) and Catalyst statistics
  (`mocap/cost/statistics.py`) into a *predicted* `PlanMetrics`. No query
  is run.
- **`mocap/cost/estimator.py`** — post-execution. Executes the candidate
  and reads Spark's own SQL execution metrics off the finished plan to
  produce ground-truth `ExecutionTelemetry`.
- **`mocap/cost/learned.py` + `mocap/calibration/`** — a linear correction
  fit on (predicted, actual) pairs, applied only to the predicted cost,
  not latency or the individual resource components.

## Known limitations (state explicitly in the paper, do not omit)

1. **CPU-seconds is an approximation**, not measured per-executor CPU
   time — it is wall-clock elapsed time × active cores. Spark's SQL
   metrics expose I/O and shuffle byte counters but not executor CPU
   time directly; a real measurement would need a `SparkListener`
   attached via py4j's callback server, which was deliberately not built
   due to interface fragility across Spark versions.
2. **The structural predictor is a transparent heuristic baseline**, not
   an empirically learned model — its coefficients (`PredictorConfig` in
   `predictor.py`) are hand-set planning priors. Only the calibration
   layer on top of it is fit from data. Do not describe the base
   predictor itself as learned.
3. **Pricing rates are illustrative**, not real cloud billing figures.

## Change log

| Date | Change |
|---|---|
| 2026-09 | Initial freeze: rates above, CPU-seconds-from-wall-clock approximation documented |
