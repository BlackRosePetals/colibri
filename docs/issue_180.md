# issue_180: Datapoint: M3 Ultra (Mac Studio, 256 GB, 80-core GPU) — 1.45 tok/s @ 82% hit, Metal + tuned flags

- **URL:** https://github.com/JustVugg/colibri/issues/180
- **State:** CLOSED (opened 2026-07-14 by NullNaveen, closed 2026-07-14)

## Summary

Only Ultra-class Apple Silicon datapoint in the tracker. Mac Studio M3 Ultra, 256 GB unified, 80-core GPU, `main` built `make glm METAL=1`, model `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`. **Shared box** (concurrent 24-user Ollama workload, ~46 GB resident on the same GPU) — numbers are a lower bound.

| Config | tok/s | expert hit | experts/token | RSS |
|---|---|---|---|---|
| CPU, `--ram 40`, defaults | 0.07 | 19% | 1021 | 37 GB |
| Metal, `--ram 40`, defaults | 0.39 | 21% | 1142 | 37 GB |
| **Metal + tuned, `--ram 160`** | **1.45** (peak 1.75) | **82%** | **381** | 131 GB |

- Tuned config: `COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 IO_THREADS=8 ./coli run --ram 160 --topp 0.7`
- Profile of the 1.45 run: expert-disk 75.3s service / 0.0s wait, expert-matmul 49.3s, attention 45.0s, other 82.1s
- Confirms `MTP=0` is a large win: 1142 -> 381 experts/token; speculation was working (53-67% acceptance) but still not worth it
- Note: 82% hit rate helped here (256 GB box, `--ram 160`) — contrast with #379's minimal-cache finding on 128 GB M5 Max

## Relevance

- **Goal a (fmt_2_m1u):** The closest hardware analog to the M1 Ultra target — Ultra-class, unified memory, same tuned flag set. Template for the M1 Ultra Config E and `--ram` exploration.
- **Goal b (fmt_4_m1u):** Context only.

## See also

- [issue_103](issue_103.md), [pr_116](pr_116.md), [issue_379](issue_379.md)
