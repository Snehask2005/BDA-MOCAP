"""Paired E3 planning benchmark: beam width 40, uncapped vs cap 48."""

import sys
import benchmark_e3_prescoring as benchmark

# Reuse the established harness and its Spark/timing configuration.
benchmark.BEAM_WIDTH = 40
benchmark.ANALYSIS_LIMITS = (None, 48)
benchmark.REPETITIONS_DEFAULT = 3

# Keep the existing benchmark artifacts untouched.
benchmark.RESULTS = benchmark.ROOT / "results/e3/beam40_cap48"

if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--all-queries",
        "--repetitions",
        "3",
    ]
    benchmark.main()
