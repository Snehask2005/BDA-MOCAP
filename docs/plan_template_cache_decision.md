# Plan-Template Cache: Engineering Decision

## Decision

**No-go for the current LRU plan-template-cache design.** Retain AST
reuse in the candidate generator and do not add another cache unless a
repeatable workload demonstrates a meaningful end-to-end benefit.

This is an engineering decision based on the measurements currently
available. It does not establish that all forms of template caching
would be ineffective.

## Existing Optimization

The candidate generator parses the input query into a SQLGlot AST once
within join-order generation and passes that tree as `tree_template`
to `_build_join_order_sql`. Each generated join order copies the template
before modifying the join structure. This avoids reparsing the original
query separately for every generated join order.

Some parsing still occurs in other stages, including table-statistics
collection and pre-scoring. These operations are distinct from repeated
per-candidate SQL-generation parsing.

## Recorded Measurements

| Approach | Median SQL-generation time |
|---|---:|
| Reparse original query for candidate generation | 62.128 ms |
| Reuse a parsed AST template | 28.809 ms |
| Uncached LRU-cache prototype | 29.027 ms |
| Cached LRU-cache prototype | 29.052 ms |

AST reuse reduced the measured SQL-generation time by approximately
53.6% relative to reparsing.

The LRU-cache prototype was approximately 0.09% slower than its uncached
comparison (a difference of 0.025 ms). This is not evidence of a useful
performance improvement and is too small to justify adding cache
complexity on its own.

These are isolated measurements, not proof that caching can never help
under a repeated-query production workload.

## Correctness and Complexity Considerations

ASTs are mutable, so a cached AST must not be modified in place when
generating a candidate. The existing copy-before-modification pattern
should be preserved.

Caching physical plans is a separate, riskier design: plans may depend
on schema, statistics, predicates, Spark configuration, and join hints.
No physical-plan cache is adopted by this decision.

## Reconsideration Criteria

Reconsider a cache only when a representative repeated-query workload
shows a repeatable end-to-end planning-time improvement beyond timing
noise, with stable results and no stale-plan or cross-query contamination.

Any future cache benchmark should compare identical workloads,
separate cache-hit and cache-miss behavior, report repeated-run medians,
and measure total candidate-generation latency rather than relying only
on a microbenchmark.

## Status

- AST reuse: implemented.
- LRU plan-template-cache prototype: evaluated, no measurable benefit.
- Production plan-template cache: not implemented.
- Follow-up: revisit only if profiling identifies a workload with
  substantial repeated-template construction cost.
