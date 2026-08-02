# issue_387: Datapoint: M5 Max 128 GB — decode-tuning A/Bs (MTP x cache 2x2): 0.4 -> 2.0 tok/s at identical output

- **URL:** https://github.com/JustVugg/colibri/issues/387
- **State:** OPEN (opened 2026-07-18 by monotophic)
- **Labels:** metal, benchmark

## Summary

Controlled, identical-output A/B campaign (greedy temp-0, md5-checked token streams, pristine `.coli_usage` snapshot restored per run) on M5 Max 128 GB, measured 2026-07-15..17, `main @ 72d3d37`, model `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp` (144 shards / 384 GB).

- **Champion config:** `MTP=0 CAP_RAISE=0 AUTOPIN=0 coli run --ram 90 --cap 1` — takes the box from ~0.4 tok/s (defaults) to ~2.0 tok/s
- **`MTP=0`:** +79% at a cached config, +10-22% at cap1; -20-45% bytes moved
- **`--cap 1` (no LRU cache):** smaller cache = faster, always; expert-matmul 20.6s -> 35.4s -> 78.4s as the cache grows
- **`--ram`:** flat on speed across 20-40 GB; choose on heat/energy, not speed
- **`COLI_MMAP=1`:** do not use on Metal — 39x slower (GPU demand-faults a 358 GB mapping)
- First two measurement rounds were invalid and discarded; the methodology section documents what survived (temp 0, usage-snapshot restore, etc.)

This is the data behind the #379 diagnosis and the PR #386 defaults change.

## Relevance

- **Goal a (fmt_2_m1u):** Direct tuning playbook for the M1 Ultra benchmark — MTP off, cache minimal, ram sizing, and a rigorous identical-output methodology worth mirroring.
- **Goal b (fmt_4_m1u):** Shared — same tuning applies to fmt=4 benchmark runs.

## See also

- [issue_379](issue_379.md), [pr_386](pr_386.md)
