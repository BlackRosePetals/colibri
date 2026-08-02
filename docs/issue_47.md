# issue_47: Benchmark: Apple M3 Max · macOS · 128 GB · internal SSD (12.5 GB/s) — 0.35–0.45 tok/s; PILOT neutral; confirms #8

- **URL:** https://github.com/JustVugg/colibri/issues/47
- **State:** CLOSED (opened 2026-07-11 by fabio-rovai, closed 2026-07-13)

## Summary

Pre-Metal CPU-era benchmark datapoint on Apple M3 Max (12P+4E), 128 GB unified, internal SSD measured at 12.49 GB/s O_DIRECT. Container: `jlnsrk/GLM-5.2-colibri-int4` (144 shards, 379 GB). Greedy `--temp 0`, `--ngen 400`, ~1k-token prompt.

| config | tok/s | expert hit | RSS |
|---|---|---|---|
| baseline, `MTP=0` | 0.35 | 68.5% | 82.2 GB |
| + `PIN PIN_GB=90` | 0.36 | 70.6% | 95.8 GB |
| + `PILOT=1` | 0.36 (neutral) | 70.6% | 95.7 GB |
| + `DRAFT=4` | 0.34 | 70.4% | 97.5 GB |
| + `--topp 0.7` | **0.45 (+29%)** | 70.0% | 96.1 GB |

- Baseline profile per 400 tok: expert-disk 400s, expert-matmul 311s, attention 351s, altro 89s — attention ~30% of token cost on CPU
- `--topp 0.7` cut expert loads 28.7 -> 17.9/layer
- Confirmed #8: int4 MTP head gives 0% acceptance (0/24) in all configs
- Also reported a setup.sh libomp false-positive

## Relevance

- **Goal a (fmt_2_m1u):** Historical baseline — the CPU-only world before Metal (#72). Documents the `--topp 0.7` lever and the MTP acceptance problem (#8) that motivated `MTP=0` in later configs.
- **Goal b (fmt_4_m1u):** Not directly relevant.

## See also

- [issue_87](issue_87.md), [issue_103](issue_103.md), [pr_72](pr_72.md)
