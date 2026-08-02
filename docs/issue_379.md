# issue_379: Metal — growing the expert cache makes decode SLOWER

- **URL:** https://github.com/JustVugg/colibri/issues/379
- **State:** OPEN (opened 2026-07-18 by monotophic)
- **Labels:** bug, metal

## Summary

On Metal (M5 Max 128 GB, macOS 26.5.2, `main @ 72d3d37`), **every expert-cache mechanism makes decode slower** — raising the LRU cap, adding a pin, or enabling AUTOPIN all reduce tok/s even as hit rate rises. Fastest config found: *minimal* cache at **~2.0 tok/s** vs ~0.4 at auto defaults (5× from config alone).

- Champion config: `MTP=0 CAP_RAISE=0 AUTOPIN=0 --cap 1`
- Suspected mechanism: GPU-side buffer residency / page-mapping cost on large, sparsely-reused host buffers registered via `newBufferWithBytesNoCopy` — on Metal, cache hits cost more than the page-cache misses they replace
- Cross-platform contrast: DGX Spark GB10 gets 2.39 tok/s at 82% hit with a large LRU + AUTOPIN; the M5 Max gets its best at **2% hit, no cache**
- Four alternative mechanisms falsified via controlled A/Bs (data in #387)
- Led directly to PR #386 (platform-aware cache defaults via storage probe)

## Relevance

- **Goal a (fmt_2_m1u):** Critical tuning context — on a 128 GB Apple Silicon box, a *big* expert cache can be anti-productive; M1 Ultra benchmark configs must A/B cache size (`--cap`, CAP_RAISE, AUTOPIN), not assume more cache = faster.
- **Goal b (fmt_4_m1u):** Shared — same cache tuning applies to the fmt=4 benchmark runs after backend fixes land.

## See also

- [issue_387](issue_387.md) — the A/B data behind this diagnosis
- [pr_386](pr_386.md) — the merged defaults change
