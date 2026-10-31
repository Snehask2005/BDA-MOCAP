# Supplementary TPC-DS-Derived Exploratory Evaluation

**Status:** Supplementary experimental record
**Relationship to Phase 4:** Additional exploratory evaluation; does not modify the primary Section 4.1 results.

## 1. Purpose and Experimental Scope

This document consolidates exploratory experiments conducted to investigate planning overhead, candidate pruning quality, execution-time variability, and candidate pre-scoring in MOCAP.

The experiments use two different local datasets, and their results must be interpreted separately:

- `benchmarks/tpcds/full_sf10/parquet`: A local, partial TPC-DS-derived dataset. It is not a complete official TPC-DS SF10 dataset.
- `benchmarks/tpcds/representative/parquet`: The representative dataset used for the five-table Q05 experiments.

The Parquet dataset directories are local experimental inputs and are not
included in this supplementary commit because of their size. Reproducing
these experiments requires obtaining matching dataset snapshots and placing
them at the paths above. The Q05 SQL query file is included separately as
a small experimental input.


The five-table experiments use the query at:

`benchmarks/tpcds/representative/queries/q5_join_order.sql`

The compared Spark configurations use `local[2]`, with adaptive query execution and cost-based optimization disabled. Automatic broadcast join selection is also disabled through `spark.sql.autoBroadcastJoinThreshold=-1`.

These experiments provide supplementary evidence and are not a replacement for the primary Phase 4 E3 evaluation.

## 2. Planning Overhead on the Local Full-SF10-Derived Dataset

The first experiment compares exhaustive candidate generation against beam search with different beam widths.

Planning time was measured independently of query execution, with three repetitions per configuration.

| Search mode | Beam width | Candidate count | Median planning time |
|---|---:|---:|---:|
| Exhaustive | — | 144 | 2.700660 s |
| Beam | 2 | 9 | 0.183155 s |
| Beam | 5 | 18 | 0.339601 s |
| Beam | 10 | 33 | 0.609302 s |
| Beam | 20 | 63 | 1.188403 s |

Beam search reduced planning overhead substantially in this five-table experiment. Increasing the beam width increased the number of generated candidates and the time spent generating them.

These are planning-only measurements. They do not establish that the fastest planning configuration also produces the best execution plan.

**Artifact files:**

- `experiments/benchmark_tpcds_full_sf10_planning.py`
- `results/tpcds_full_sf10_planning_runs.csv`
- `results/tpcds_full_sf10_planning_summary.csv`

## 3. Initial Execution-Quality Comparison

A separate experiment compared the execution quality of exhaustive candidate generation and beam search on the local full-SF10-derived dataset.

Each candidate was initially screened with one execution measurement. The results were checked for output equivalence.

| Search mode | Total candidates | Successful equivalent candidates | Best observed execution time | Observed loss |
|---|---:|---:|---:|---:|
| Exhaustive | 144 | 48 | 24.521921 s | 0.00% |
| Beam 2 | 9 | 3 | 32.254248 s | 31.5323% |
| Beam 5 | 18 | 6 | 32.206810 s | 31.3389% |
| Beam 10 | 33 | 11 | 27.329558 s | 11.4495% |
| Beam 20 | 63 | 21 | 26.286833 s | 7.1973% |

The exhaustive-best fingerprint was `768bc938304bd152`. It was not retained by any of the four beam configurations in this initial comparison.

The observed loss was calculated from the fastest observed execution time in each candidate set relative to the exhaustive-best observed execution time.

**Important limitation:** These measurements used a single execution per candidate. The reported percentages are preliminary observations and should not be interpreted as stable estimates of pruning loss.

**Artifact files:**

- `experiments/benchmark_tpcds_full_sf10_execution_quality.py`
- `results/tpcds_full_sf10_execution_quality_runs.csv`
- `results/tpcds_full_sf10_execution_quality_summary.csv`

## 4. Repeated Execution of Shortlisted Finalists

Two follow-up experiments examined whether the initial execution-time rankings remained stable under repeated measurements.

These experiments did not repeat every candidate. They shortlisted candidates based on the initial single-run screening results and then measured only those finalists.

### 4.1 Top-two finalist experiment

The top two candidates from each search configuration were measured three additional times.

| Search mode | Best repeated median | Observed difference relative to repeated exhaustive reference |
|---|---:|---:|
| Exhaustive | 32.969373 s | 0.0000% |
| Beam 2 | 36.866613 s | 11.8208% |
| Beam 5 | 39.431861 s | 19.6015% |
| Beam 10 | 33.725908 s | 2.2947% |
| Beam 20 | 31.441856 s | -4.6331% |

### 4.2 Top-one winner experiment

A second experiment selected one initial winner per configuration and measured each selected candidate five times.

| Search mode | Best repeated median | Observed difference relative to repeated exhaustive reference |
|---|---:|---:|
| Exhaustive | 34.718652 s | 0.0000% |
| Beam 2 | 40.597886 s | 16.9339% |
| Beam 5 | 41.246933 s | 18.8034% |
| Beam 10 | 32.520169 s | -6.3323% |
| Beam 20 | 35.214385 s | 1.4279% |

Negative values mean that the selected beam finalist had a lower repeated median than the repeated timing of the initial exhaustive winner.

These comparisons demonstrate that single-run execution rankings can change after repeated measurements.

However, neither experiment remeasured the complete exhaustive candidate set. Therefore, a beam finalist outperforming the repeated timing of the initial exhaustive winner does not establish that beam search found a better plan than the true best exhaustively generated plan.

**Artifact files:**

- `experiments/repeat_tpcds_full_sf10_finalists.py`
- `experiments/repeat_tpcds_full_sf10_winners.py`
- `results/tpcds_full_sf10_finalist_repeat_candidates.csv`
- `results/tpcds_full_sf10_finalist_repeat_runs.csv`
- `results/tpcds_full_sf10_finalist_repeat_summary.csv`
- `results/tpcds_full_sf10_winner_repeat_candidates.csv`
- `results/tpcds_full_sf10_winner_repeat_runs.csv`
- `results/tpcds_full_sf10_winner_repeat_summary.csv`

## 5. Pooled Q05 Broadcast-Candidate Quality Comparison

Two experiments were conducted on the representative five-table Q05 dataset:

1. Beam-width sweep.
2. Broadcast-only pruning evaluation.

The experiments used the same query, dataset, Spark configuration, search seed, and candidate fingerprints. Both used one warm-up execution per candidate, followed by five measured executions. Their candidate sets were identical:

| Candidate set | Exhaustive | Beam width 10 |
|---|---:|---:|
| Total generated candidates | 144 | 33 |
| Broadcast candidates | 48 | 11 |

The two experiments therefore provide ten recorded execution measurements per broadcast candidate when their raw measurements are pooled.

The combined measurements produced the following result:

| Metric | Exhaustive | Beam width 10 |
|---|---:|---:|
| Broadcast candidates measured | 48 | 11 |
| Measurements per candidate | 10 | 10 |
| Best candidate fingerprint | `996bd36b9b0504cd` | `7c704888ab279388` |
| Best candidate median | 0.420490 s | 0.422049 s |

The combined observed pruning loss is:

\[
\left(\frac{0.422049}{0.420490}-1\right)\times100
=0.3708\%
\]

The exhaustive-best physical plan was not retained by beam width 10. Nevertheless, the best retained beam candidate had an observed median execution time only 0.3708% higher than the exhaustive-best candidate's pooled median.

This is a useful supplementary result: in this particular Q05 representative-data experiment, beam width 10 substantially reduced the broadcast-candidate set while preserving a candidate with nearly the same observed execution performance as the exhaustive-best candidate.

The result is based on pooled measurements from two separate benchmark runs, not one continuous experiment. Although the candidate sets and principal Spark settings matched, it should be treated as an exploratory estimate rather than a general pruning-loss guarantee.

**Artifact files:**

- `experiments/measure_tpcds_beam_width_sweep.py`
- `experiments/measure_tpcds_pruning_loss.py`
- `results/tpcds_beam_width_sweep_runs.csv`
- `results/tpcds_beam_width_sweep_summary.csv`
- `results/tpcds_broadcast_pruning_runs.csv`
- `results/tpcds_broadcast_pruning_summary.csv`

## 6. Candidate Pre-Scoring Evaluation

### 6.1 Q05 representative-data experiment

The Q05 pre-scoring experiment investigated reducing the number of physical plans selected for expensive analysis.

| Configuration | Unique physical plans |
|---|---:|
| Baseline | 33 |
| Pre-scoring limit 12 | 9 |
| Pre-scoring limit 20 | 15 |

The top three finalists per configuration were selected and measured five times. All screened results were equivalent, with no recorded execution errors in this experiment.

The best repeated finalist medians were:

| Configuration | Best finalist median |
|---|---:|
| Baseline | 0.135415 s |
| Pre-scoring limit 12 | 0.170323 s |
| Pre-scoring limit 20 | 0.170323 s |

The summary's `latency_change_percent_vs_baseline` uses the repeated median of the baseline candidate selected during the original single-run screening as its reference. It does not use the baseline row's best finalist median as the denominator.

Consequently, the summary's percentage and a direct comparison of the displayed best finalist medians answer different questions. They should not be mixed or presented as the same comparison.

The experiment demonstrates a reduction in the number of physical plans considered, but it does not establish an execution-quality improvement from pre-scoring.

**Artifact files:**

- `experiments/benchmark_e3_prescoring_quality_q05.py`
- `results/e3/prescoring_q05_quality_runs.csv`
- `results/e3/prescoring_q05_quality_summary.csv`

### 6.2 Q10 pre-scoring experiment

The available Q10 pre-scoring result file contains 15 baseline screening measurements. Two measurements recorded errors, and the file does not contain corresponding pre-scoring-limit runs.

Therefore, this experiment is incomplete and is excluded from pre-scoring quality conclusions.

**Artifact files:**

- `experiments/benchmark_e3_prescoring_quality.py`
- `results/e3/prescoring_q10_quality_runs.csv`

## 7. Overall Interpretation

The supplementary experiments support the following observations:

1. Beam search reduces candidate-generation overhead compared with exhaustive generation in the tested five-table setup.
2. Smaller beam widths can reduce planning time substantially, but initial execution-quality measurements indicate a potential trade-off.
3. Repeated execution measurements can change candidate rankings, making one-run execution comparisons unreliable as final quality estimates.
4. The pooled Q05 broadcast-candidate experiment produced an observed pruning loss of 0.3708%, despite the exhaustive-best physical plan not being retained.
5. Pre-scoring reduces the number of physical plans considered in Q05, but the available results do not establish an execution-quality improvement.
6. The available Q10 pre-scoring measurements are incomplete and should not be used to support a quality claim.

## 8. Limitations

- The local full-SF10-derived dataset is partial and must not be presented as the complete official TPC-DS SF10 benchmark.
- The full-data planning and execution experiments use a five-table query; their findings should not be generalized to all query sizes.
- Initial execution-quality measurements use one execution per candidate.
- Repeated finalist experiments evaluate shortlisted candidates rather than the complete exhaustive candidate set.
- The pooled Q05 broadcast comparison is limited to broadcast candidates and one representative five-table query.
- The Q05 pre-scoring summary uses a screened baseline reference that differs from the baseline row's best repeated finalist median.
- The Q10 pre-scoring experiment does not contain sufficient pre-scoring measurements for a comparison.

## 9. Relationship to the Main Phase 4 Results

This supplementary evaluation does not replace or modify the merged results documented in:

`docs/section_4_1_scalable_candidate_search.md`

The primary Phase 4 results retain their original experimental scope and conclusions. This document preserves additional exploratory measurements, including their limitations, for future research and reproducibility.

No general exhaustive-optimum pruning-loss guarantee should be inferred from these supplementary experiments.
