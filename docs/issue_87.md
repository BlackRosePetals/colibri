# issue_87: [Performance]: M5 MAX 128GB with METAL - 1.83 tok/s

- **URL:** https://github.com/JustVugg/colibri/issues/87
- **State:** CLOSED (opened 2026-07-12 by baha2046, closed 2026-07-13)

## Summary

First M5 Max Metal datapoint, using the pre-merge `codearranger/colibri` `metal-backend` branch (what became PR #72) with early Fable5 tuning: `DIRECT=1 IO_THREADS=8`, MTP off.

- Hardware: MacBook M5 Max 128 GB, 2 TB
- iobench on an expert shard: buffered 17.35 GB/s; O_DIRECT 64.68 GB/s (8 threads, 19 MB blocks)
- Result: **1.83 tok/s** with `COLI_MODEL=... ./coli run "Compare the myths of Lucifer and Prometheus" --ram 96`
- Log shows the auto macOS banner: "MTP ... su macOS il decode liscio e' piu' veloce (A/B misurato 1.68 vs 0.85 tok/s): draft OFF"
- Same Lucifer/Prometheus prompt later used by #103 and the M5 Max report

## Relevance

- **Goal a (fmt_2_m1u):** Part of the result lineage: 1.83 (#87) -> 2.06 (#103) -> 2.24 (PR #116 report). Establishes the prompt and the `DIRECT=1 IO_THREADS=8 MTP=0` lever set.
- **Goal b (fmt_4_m1u):** Context only.

## See also

- [issue_103](issue_103.md), [pr_72](pr_72.md), [pr_116](pr_116.md)
