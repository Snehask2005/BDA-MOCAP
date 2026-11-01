"""
Experiment harness: measure once, evaluate every strategy (owner: Anaswara).

Protocol
--------
1. For each workload query, Sneha's generator produces the candidate pool
   (beam search + pre-scoring + parallel Catalyst analysis).
2. Every candidate gets a pre-execution prediction (predictor is pluggable;
   default = mocap.cost.predictor with candidate-specific Catalyst stats, so
   Hridhika's learned model can be swapped in with ``predictor=``).
3. Every candidate is EXECUTED ``repeats`` times through execution.run_plan,
   so its true latency/cost distribution is known. Spark's own planner is
   also run on the original SQL with CBO/AQE on and off.
4. All selection strategies (execution.baselines) then choose from the same
   measured pool, and their choice is scored with the measured runs. This
   makes the comparison paired per query and lets E2's budget sweep and the
   robustness suite reuse the same measurements without re-running Spark.

Conformal calibration is query-grouped: the quantile used for query q is
computed only from candidates of OTHER queries (leave-one-query-out), so
no run of q ever leaks into its own calibration set.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from execution import baselines as B
from execution.config import Env
from execution.run_plan import PlanRunner, RunSpec
from execution.stats import median, percentile
from execution.workload import WorkloadQuery

Predictor = Callable[[Any, Any], Tuple[float, float]]
Perturb = Callable[[str, str, float, float], Tuple[float, float]]

SPARK_DEFAULT_REFERENCE = "spark_default_cbo_off_aqe_on"  # Spark 4.x out-of-the-box


@dataclass
class CandidateRecord:
    plan_id: str
    strategy: str
    kind: str = "candidate"            # "candidate" | "spark_default"
    fingerprint: Optional[str] = None
    sql: Optional[str] = None
    pred_cost: Optional[float] = None
    pred_latency: Optional[float] = None
    latencies_s: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    executor_seconds: List[float] = field(default_factory=list)
    bytes_scanned: List[float] = field(default_factory=list)
    bytes_shuffled: List[float] = field(default_factory=list)
    failed_runs: int = 0
    plan_verified: Optional[bool] = None
    num_broadcast_joins: int = 0
    num_sort_merge_joins: int = 0
    num_shuffle_hash_joins: int = 0

    @property
    def actual_latency(self) -> Optional[float]:
        return median(self.latencies_s) if self.latencies_s else None

    @property
    def actual_cost(self) -> Optional[float]:
        return median(self.costs) if self.costs else None


@dataclass
class QueryMeasurement:
    query_id: str
    family: str
    num_tables: int
    selectivity: Optional[str]
    candidates: List[CandidateRecord] = field(default_factory=list)
    spark_defaults: List[CandidateRecord] = field(default_factory=list)
    candidates_generated: int = 0
    planning_seconds: float = 0.0
    prediction_seconds: float = 0.0
    error: Optional[str] = None
    scenario: str = "baseline"
    meta: Dict[str, Any] = field(default_factory=dict)

    def measured_pool(self) -> List[CandidateRecord]:
        return [c for c in self.candidates if c.actual_latency is not None]

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=1)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryMeasurement":
        data = dict(data)
        data["candidates"] = [CandidateRecord(**c) for c in data.get("candidates", [])]
        data["spark_defaults"] = [CandidateRecord(**c) for c in data.get("spark_defaults", [])]
        return cls(**data)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def default_predictor(spark, candidate) -> Tuple[float, float]:
    """Pre-execution prediction with candidate-specific Catalyst statistics."""

    from mocap.cost.predictor import predict_candidates
    from mocap.cost.statistics import extract_plan_statistics

    statistics = extract_plan_statistics(spark.sql(candidate.sql))
    metrics = predict_candidates([candidate], statistics=statistics)[candidate.plan_id]
    return float(metrics.estimated_cost), float(metrics.estimated_latency)


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------

def _record_from_rows(record: CandidateRecord, rows: Sequence[Dict[str, Any]]) -> CandidateRecord:
    verified: List[bool] = []
    for row in rows:
        if row["status"] != "ok":
            record.failed_runs += 1
            continue
        record.latencies_s.append(row["latency_ms"] / 1000.0)
        record.costs.append(row["actual_cost"])
        record.executor_seconds.append(row["executor_seconds"])
        record.bytes_scanned.append(row["bytes_scanned"])
        record.bytes_shuffled.append(row["bytes_shuffled"])
        if row.get("plan_verified") is not None:
            verified.append(bool(row["plan_verified"]))
    record.plan_verified = all(verified) if verified else None
    return record


def spark_default_overrides(env: Env) -> List[Tuple[str, Dict[str, str]]]:
    threshold = str(env.experiment.get("spark_default_broadcast_threshold", "10485760"))
    variants = []
    for v in env.experiment.get("spark_default_variants", []):
        cbo = "true" if v["cbo"] else "false"
        variants.append((
            v["name"],
            {
                "spark.sql.cbo.enabled": cbo,
                "spark.sql.cbo.joinReorder.enabled": cbo,
                "spark.sql.adaptive.enabled": "true" if v["aqe"] else "false",
                "spark.sql.autoBroadcastJoinThreshold": threshold,
            },
        ))
    return variants


def measure_query(
    spark,
    runner: PlanRunner,
    query: WorkloadQuery,
    env: Env,
    predictor: Optional[Predictor] = None,
    repeats: Optional[int] = None,
    warmup_runs: Optional[int] = None,
    max_candidates: Optional[int] = None,
    include_spark_defaults: bool = True,
    scenario: str = "baseline",
    log: Callable[[str], None] = print,
) -> QueryMeasurement:
    from mocap.plans.generator import generate_join_candidates
    from mocap.query.parser import QueryRequest

    exp = env.experiment
    predictor = predictor or default_predictor
    max_candidates = int(max_candidates or exp.get("max_candidates_measured", 48))
    m = QueryMeasurement(query.query_id, query.family, query.num_tables, query.selectivity,
                         scenario=scenario)

    request = QueryRequest(query_id=query.query_id, sql=query.sql, budget=1.0)
    search_mode = exp.get("search_mode", "beam")
    kwargs: Dict[str, Any] = {
        "search_mode": search_mode,
        "beam_width": int(exp.get("beam_width", 40)),
        "candidate_workers": int(exp.get("candidate_workers", 1)),
    }
    if search_mode == "beam" and exp.get("candidate_analysis_limit"):
        kwargs["candidate_analysis_limit"] = int(exp["candidate_analysis_limit"])

    start = time.perf_counter()
    try:
        candidates = generate_join_candidates(request, spark, **kwargs)
    except Exception as exc:
        m.error = f"candidate generation failed: {exc}"
        log(f"  [{query.query_id}] {m.error}")
        return m
    m.planning_seconds = time.perf_counter() - start
    m.candidates_generated = len(candidates)
    if not candidates:
        m.error = "generator returned no candidates"
        return m
    candidates = candidates[:max_candidates]

    start = time.perf_counter()
    for cand in candidates:
        rec = CandidateRecord(
            plan_id=cand.plan_id,
            strategy=cand.strategy,
            fingerprint=cand.fingerprint,
            sql=cand.sql,
            num_broadcast_joins=cand.num_broadcast_joins,
            num_sort_merge_joins=cand.num_sort_merge_joins,
            num_shuffle_hash_joins=cand.num_shuffle_hash_joins,
        )
        try:
            rec.pred_cost, rec.pred_latency = predictor(spark, cand)
        except Exception as exc:
            log(f"  [{query.query_id}] prediction failed for {cand.plan_id}: {exc}")
        m.candidates.append(rec)
    m.prediction_seconds = time.perf_counter() - start

    for i, (cand, rec) in enumerate(zip(candidates, m.candidates), start=1):
        rows = runner.run(
            RunSpec(query.query_id, cand.plan_id, cand.strategy, cand.sql,
                    expected_fingerprint=cand.fingerprint),
            repeats=repeats, warmup_runs=warmup_runs,
        )
        _record_from_rows(rec, rows)
        log(f"  [{query.query_id}] {i}/{len(candidates)} {cand.plan_id} "
            f"median={rec.actual_latency if rec.actual_latency is not None else float('nan'):.3f}s")

    if include_spark_defaults:
        for name, overrides in spark_default_overrides(env):
            rec = CandidateRecord(plan_id=f"{query.query_id}_{name}", strategy=name,
                                  kind="spark_default", sql=query.sql)
            rows = runner.run(
                RunSpec(query.query_id, rec.plan_id, name, query.sql, conf_overrides=overrides),
                repeats=repeats, warmup_runs=warmup_runs,
            )
            _record_from_rows(rec, rows)
            m.spark_defaults.append(rec)
    return m


def measure_workload(
    spark,
    runner: PlanRunner,
    queries: Sequence[WorkloadQuery],
    env: Env,
    cache_dir: Optional[Path] = None,
    reuse: bool = True,
    log: Callable[[str], None] = print,
    **kwargs,
) -> List[QueryMeasurement]:
    """Measure every query, caching each to ``cache_dir/<query_id>.json``."""

    results: List[QueryMeasurement] = []
    for n, query in enumerate(queries, start=1):
        path = Path(cache_dir) / f"{query.query_id}.json" if cache_dir else None
        if path is not None and reuse and path.exists():
            results.append(load_measurement(path))
            log(f"[{n}/{len(queries)}] {query.query_id}: reused cached measurement")
            continue
        log(f"[{n}/{len(queries)}] {query.query_id}: measuring ({query.num_tables} tables)")
        m = measure_query(spark, runner, query, env, log=log, **kwargs)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(m.to_json(), encoding="utf-8")
        results.append(m)
    return results


def load_measurement(path: Path) -> QueryMeasurement:
    return QueryMeasurement.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_measurements(cache_dir: Path) -> List[QueryMeasurement]:
    return [load_measurement(p) for p in sorted(Path(cache_dir).glob("*.json"))]


@contextlib.contextmanager
def background_load(spark, sql: str, n_jobs: int) -> Iterator[None]:
    """Run ``n_jobs`` copies of ``sql`` continuously in background threads."""

    stop = threading.Event()

    def worker(idx: int) -> None:
        spark.sparkContext.setJobGroup(f"mocap-bgload-{idx}", "background load", interruptOnCancel=True)
        while not stop.is_set():
            try:
                spark.sql(sql).write.format("noop").mode("overwrite").save()
            except Exception:
                if stop.is_set():
                    break
                time.sleep(0.5)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_jobs)]
    for t in threads:
        t.start()
    time.sleep(1.0)  # let the load ramp up
    try:
        yield
    finally:
        stop.set()
        for i in range(n_jobs):
            try:
                spark.sparkContext.cancelJobGroup(f"mocap-bgload-{i}")
            except Exception:
                pass
        for t in threads:
            t.join(timeout=30)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

def calibration_samples(measurements: Sequence[QueryMeasurement], perturb: Optional[Perturb] = None):
    from selector.conformal import CalibrationSample

    samples = []
    for m in measurements:
        for c in m.measured_pool():
            if not c.pred_cost or not c.pred_latency or not c.actual_cost or not c.actual_latency:
                continue
            pc, pl = c.pred_cost, c.pred_latency
            if perturb is not None:
                pc, pl = perturb(m.query_id, c.plan_id, pc, pl)
            samples.append(CalibrationSample(
                query_id=m.query_id, plan_id=c.plan_id,
                predicted_cost=pc, actual_cost=c.actual_cost,
                predicted_latency=pl, actual_latency=c.actual_latency,
            ))
    return samples


# Conformal score used for calibration:
#   "signed"   s_i = log(C_i / C_hat_i)    one-sided upper bound C_hat * exp(q)
#   "absolute" s_i = |log(C_i / C_hat_i)|  Roshini's selector default
# Both are split-conformal with the same finite-sample rank
# k = ceil((n+1)(1-delta)), so both give P(C <= C_hat * exp(q)) >= 1 - delta
# under exchangeability. The signed score is the natural one for a ONE-sided
# budget constraint and, unlike |.|, it can shrink a systematically
# over-estimating predictor (exp(q) < 1), so the budget filter does not
# reject every plan when the predictor's scale is off.
RESIDUAL_MODE = "signed"


def set_residual_mode(mode: str) -> None:
    global RESIDUAL_MODE
    if mode not in ("signed", "absolute"):
        raise ValueError("conformal residual mode must be 'signed' or 'absolute'")
    RESIDUAL_MODE = mode



def _signed_quantile(
    pred: Sequence[float],
    actual: Sequence[float],
    delta: float,
    metric: str,
):
    from selector.conformal import ConformalQuantile

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be between 0 and 1")

    if len(pred) != len(actual):
        raise ValueError("pred and actual must have equal lengths")

    if not pred:
        raise ValueError("calibration samples cannot be empty")

    scores = sorted(
        math.log(max(a, 1e-300)) - math.log(max(p, 1e-300))
        for p, a in zip(pred, actual)
    )

    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - delta))

    if rank > n:
        q = math.inf
        idx = n
    else:
        idx = rank - 1
        q = scores[idx]

    multiplier = math.exp(q) if math.isfinite(q) else math.inf

    return ConformalQuantile(
        metric_name=metric,
        delta=delta,
        confidence_level=1.0 - delta,
        sample_size=n,
        quantile_index=idx,
        raw_quantile_q=float(q),
        multiplier=float(multiplier),
        residuals_summary={
            "min": scores[0],
            "median": scores[n // 2],
            "max": scores[-1],
            "q_val": float(q),
            "mode": "signed",
            "rank_exceeds_n": rank > n,
        },
    )

def calibrate(samples, delta: float, mode: Optional[str] = None):
    from selector.conformal import ConformalCalibrator

    if not samples:
        raise ValueError("no calibration samples")
    mode = mode or RESIDUAL_MODE
    if mode == "absolute":
        return ConformalCalibrator().calibrate_joint(samples, delta_cost=delta, delta_latency=delta)
    return (
        _signed_quantile([s.predicted_cost for s in samples], [s.actual_cost for s in samples], delta, "cost"),
        _signed_quantile([s.predicted_latency for s in samples], [s.actual_latency for s in samples],
                         delta, "latency"),
    )


def loqo_quantiles(
    measurements: Sequence[QueryMeasurement],
    delta: float,
    perturb: Optional[Perturb] = None,
) -> Dict[str, Tuple[Any, Any]]:
    """Leave-one-query-out conformal quantiles for every query."""

    per_query = {m.query_id: calibration_samples([m], perturb) for m in measurements}
    out: Dict[str, Tuple[Any, Any]] = {}
    for qid in per_query:
        others = [s for other, ss in per_query.items() if other != qid for s in ss]
        if others:
            out[qid] = calibrate(others, delta)
    return out


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def _outcome(
    m: QueryMeasurement,
    choice: B.Choice,
    record: Optional[CandidateRecord],
    budget: float,
    multiplier: float,
    delta: float,
    oracle_rec: Optional[CandidateRecord],
    reference: Optional[CandidateRecord],
    pred: Tuple[Optional[float], Optional[float]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "query_id": m.query_id,
        "family": m.family,
        "num_tables": m.num_tables,
        "selectivity": m.selectivity,
        "scenario": m.scenario,
        "strategy": choice.strategy,
        "budget_multiplier": multiplier,
        "delta": delta,
        "budget": budget,
        "plan_id": choice.plan_id,
        "plan_strategy": record.strategy if record else None,
        "infeasible_fallback": bool(choice.infeasible_fallback),
        "reason": choice.reason,
        "planning_ms": choice.planning_ms,
        "pred_cost": pred[0],
        "pred_latency": pred[1],
    }
    if record is None or record.actual_latency is None:
        row.update({"plan_id": None, "latency_s": None, "cost": None, "run_latencies_s": [],
                    "budget_violated": None, "run_violation_rate": None, "matches_oracle": None,
                    "oracle_regret": None, "speedup_vs_spark_default": None})
        return row
    latency, cost = record.actual_latency, record.actual_cost
    row.update({
        "latency_s": latency,
        "cost": cost,
        "p90_latency_s": percentile(record.latencies_s, 90),
        "run_latencies_s": list(record.latencies_s),
        "budget_violated": bool(cost > budget),
        "run_violation_rate": sum(c > budget for c in record.costs) / len(record.costs),
        "matches_oracle": bool(oracle_rec is not None and record.plan_id == oracle_rec.plan_id),
        "oracle_latency_s": oracle_rec.actual_latency if oracle_rec else None,
        "oracle_regret": (latency / oracle_rec.actual_latency - 1.0)
        if oracle_rec and oracle_rec.actual_latency else None,
        "speedup_vs_spark_default": (reference.actual_latency / latency)
        if reference and reference.actual_latency and latency else None,
        "plan_verified": record.plan_verified,
    })
    return row


def evaluate_query(
    m: QueryMeasurement,
    multiplier: float,
    delta: float,
    quantiles: Optional[Tuple[Any, Any]],
    seed: int,
    perturb: Optional[Perturb] = None,
    strategies: Sequence[str] = B.POOL_STRATEGIES,
    include_spark_defaults: bool = True,
    reference_name: str = SPARK_DEFAULT_REFERENCE,
) -> List[Dict[str, Any]]:
    """Score every strategy on one measured query at budget multiplier m."""

    pool = m.measured_pool()
    if not pool:
        return []
    # Perturbed predictions are what predicted-only strategies see.
    view: List[CandidateRecord] = []
    for c in pool:
        pc, pl = c.pred_cost, c.pred_latency
        if perturb is not None and pc is not None and pl is not None:
            pc, pl = perturb(m.query_id, c.plan_id, pc, pl)
        view.append(CandidateRecord(**{**asdict(c), "pred_cost": pc, "pred_latency": pl}))
    by_id = {c.plan_id: c for c in view}

    cheapest = min(c.actual_cost for c in pool)
    budget = multiplier * cheapest
    oracle_choice = B.oracle(view, budget)
    oracle_rec = by_id.get(oracle_choice.plan_id)
    reference = next((d for d in m.spark_defaults if d.strategy == reference_name), None)

    choices: List[B.Choice] = []
    for name in strategies:
        if name == "oracle":
            choices.append(oracle_choice)
        elif name == "random":
            choices.append(B.random_choice(view, budget, seed, m.query_id))
        elif name == "greedy_cheapest":
            choices.append(B.greedy_cheapest(view, budget))
        elif name == "greedy_fastest":
            choices.append(B.greedy_fastest(view, budget))
        elif name in ("mocap", "mocap_point"):
            if quantiles is None:
                continue
            choices.append(B.mocap_select(view, budget, m.query_id, quantiles[0], quantiles[1],
                                          use_conformal=(name == "mocap")))
        else:
            raise ValueError(f"unknown strategy {name}")

    rows = []
    for ch in choices:
        rec = by_id.get(ch.plan_id) if ch.plan_id else None
        pred = (rec.pred_cost, rec.pred_latency) if rec else (None, None)
        rows.append(_outcome(m, ch, rec, budget, multiplier, delta, oracle_rec, reference, pred))
    if include_spark_defaults:
        for d in m.spark_defaults:
            if d.actual_latency is None:
                continue
            ch = B.Choice(d.strategy, d.plan_id, reason="Spark's own planner on original SQL")
            rows.append(_outcome(m, ch, d, budget, multiplier, delta, oracle_rec, reference, (None, None)))
    return rows


def evaluate(
    measurements: Sequence[QueryMeasurement],
    multipliers: Sequence[float],
    deltas: Sequence[float],
    seed: int,
    perturb: Optional[Perturb] = None,
    quantile_source: Optional[Sequence[QueryMeasurement]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Evaluate all strategies for every (query, multiplier, delta).

    quantile_source=None  -> leave-one-query-out calibration on ``measurements``
                             (``perturb`` applied to calibration too)
    quantile_source=[...] -> one calibration from a separate, UNPERTURBED
                             measurement set (robustness: calibrate on the
                             clean world, test on the stressed one)
    """

    rows: List[Dict[str, Any]] = []
    for delta in deltas:
        if quantile_source is None:
            q_by_query = loqo_quantiles(measurements, delta, perturb)
        else:
            shared = calibrate(calibration_samples(quantile_source), delta)
            q_by_query = {m.query_id: shared for m in measurements}
        for m in measurements:
            for mult in multipliers:
                rows.extend(evaluate_query(m, mult, delta, q_by_query.get(m.query_id), seed,
                                           perturb=perturb, **kwargs))
    return rows


def coverage_report(measurements: Sequence[QueryMeasurement], delta: float,
                    perturb: Optional[Perturb] = None) -> Dict[str, Any]:
    """Empirical LOQO coverage of the conformal cost bound across all plans."""

    q_by_query = loqo_quantiles(measurements, delta, perturb)
    covered = total = point_covered = 0
    for m in measurements:
        if m.query_id not in q_by_query:
            continue
        cost_q = q_by_query[m.query_id][0]
        for s in calibration_samples([m], perturb):
            total += 1
            covered += s.actual_cost <= s.predicted_cost * cost_q.multiplier
            point_covered += s.actual_cost <= s.predicted_cost
    return {
        "delta": delta,
        "target_coverage": 1 - delta,
        "plans": total,
        "conformal_coverage": covered / total if total else math.nan,
        "point_coverage": point_covered / total if total else math.nan,
    }


def json_safe(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten list fields for CSV output."""

    out = {}
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out
