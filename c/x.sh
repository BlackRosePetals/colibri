#!/bin/bash -x

# ============================================================
# M1 Ultra Metal benchmark — "Compare the myths of Lucifer and Prometheus"
#
# The original run (below, Config B from METAL-M5MAX-PERF-REPORT.md)
# produces ~0.14 tok/s because PIPE defaults to 0 on macOS and OMP
# active-spin throttles the GPU.  See docs/METAL-M5MAX-PERF-REPORT.md
# for the full analysis.
#
# The winning config (E) from the M5 Max report achieves 2.24 tok/s
# by adding PIPE=1 PIPE_WORKERS=8 (async expert loading) and
# COLI_NO_OMP_TUNE=1 (passive OMP waits, preserves GPU clocks).
# M1 Ultra (24 GPU cores vs M5 Max's 40) should target >=2.0 tok/s.
# ============================================================

# OLD — missing PIPE + NO_OMP_TUNE; produces ~1.25 tok/s or worse
# COLI_METAL=1 DIRECT=1 MTP=0 ./coli run --model /Users/joe/mlx-models/GLM-5.2-colibri-int4-g64-with-int8-mtp/ --ram 110 "Compare the myths of Lucifer and Prometheus"

# SHORT RUN — verify PIPE+NO_OMP_TUNE fix with 128 tokens (fast check, ~60-120s at 0.16 tok/s or ~60s at 2 tok/s)
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 ./coli run --model /Users/joe/mlx-models/GLM-5.2-colibri-int4-g64-with-int8-mtp/ --ram 110 --ngen 128 "Compare the myths of Lucifer and Prometheus"

# FULL RUN — 1024 tokens once short run confirms correct throughput
# COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 ./coli run --model /Users/joe/mlx-models/GLM-5.2-colibri-int4-g64-with-int8-mtp/ --ram 110 "Compare the myths of Lucifer and Prometheus"
