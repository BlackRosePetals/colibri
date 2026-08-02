# issue_707: [Bug]: OpenMP spin-wait tuning is inert on macOS — and applying it costs 2.2x decode on a 32 GB host

- **URL:** https://github.com/JustVugg/colibri/issues/707
- **State:** OPEN (opened 2026-07-30 by fredchu)
- **Labels:** bug, help wanted, performance, metal

## Summary

The OpenMP hot-thread tuning block in `main()` never reaches the OpenMP runtime on macOS, because the self-re-exec that makes it effective is compiled only for Linux and FreeBSD. On the reporter's M1 Max 32 GB host that is **load-bearing in the good direction**: exporting the same four variables externally (`OMP_WAIT_POLICY=active KMP_BLOCKTIME=200 OMP_DYNAMIC=FALSE GOMP_SPINCOUNT=200000`) makes decode **2.2x slower**.

- Item 1: the tuning is inert on macOS — and that is currently a good thing
- Item 2: also about how the OpenMP team is configured on macOS (see issue body)
- The report is less "fix the missing branch" and more "don't fix it without gating — here is the measurement"
- Environment: macOS 26.5.2, M1 Max (8P+2E), 32 GB, `make colibri METAL=1`, model `mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`

## Relevance

- **Goal a (fmt_2_m1u):** Directly justifies `COLI_NO_OMP_TUNE=1` in benchmark Configs D/E (from the M5 Max report): active-spin costs GPU clocks on Apple's shared SoC power budget. M1 Ultra configs must keep OMP waits passive.
- **Goal b (fmt_4_m1u):** Shared — same flag discipline for fmt=4 runs.

## See also

- [pr_116](pr_116.md) — the M5 Max report that identified the OMP-spin regression
