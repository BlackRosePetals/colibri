# issue_107: Benchmark: Mac Mini M4 Pro (48 GB) — 0.30 tok/s with Metal backend (PR #72), 0.18 CPU-only

- **URL:** https://github.com/JustVugg/colibri/issues/107
- **State:** CLOSED (opened 2026-07-13 by bloxi-bot, closed 2026-07-13)

## Summary

First field report of the Metal backend (PR #72 branch) on a small-RAM Mac. Mac Mini M4 Pro (10P+4E CPU, 16-core GPU, 48 GB unified, ~273 GB/s), internal 1 TB SSD. Model: pre-converted int4 from `jlnsrk/GLM-5.2-colibri-int4` with **MTP heads swapped for the int8 ones** from `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp` (the main repo then still shipped the broken int4 sizes from #8).

- Disk: buffered 8.38 GB/s, F_NOCACHE 6.59 GB/s
- **Metal (PR #72), defaults, 128 tok: 0.30 tok/s** vs CPU-only 0.18 tok/s (all runs `--ram 38`)
- Metal run detail: `METAL: blocchi GPU 9018 | fallback CPU 0 | expert su GPU 100933`; expert-matmul 78.7s vs ~204s CPU-equivalent (~2.6x); METAL-ATTN 4368 layers on GPU; RSS 27.4 GB; profile 59% disk
- Throughput warmed from 0.23 at t=16 to steady 0.30-0.31 by t=64

## Relevance

- **Goal a (fmt_2_m1u):** Shows the Metal dispatch counters (`blocchi GPU`, `fallback CPU`, `expert su GPU`, `METAL-ATTN`) of a healthy run on a constrained box — the same counters the M1 Ultra fmt=4 run showed as all-zero (see [metal_dispatch_gap.md](metal_dispatch_gap.md)).
- **Goal b (fmt_4_m1u):** Context — what healthy Metal dispatch output looks like.

## See also

- [pr_72](pr_72.md), [issue_103](issue_103.md)
