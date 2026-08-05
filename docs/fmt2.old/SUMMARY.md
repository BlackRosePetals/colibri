# fmt=2 benchmark summary
generated: 2026-08-04T20:35:54Z

> Everything below the results table is hand-maintained analysis. Re-running
> `benchmark_run.sh` regenerates this file and will overwrite it.
>
> `07_Epipe16` stopped at **970 tokens** (early EOS): its tok/s is a valid rate,
> its decode seconds are not comparable to the 1024-token runs.

| run | tok/s | decode s | hit % | disk wait s | matmul s | attn s | attn gpu-sched s | expert gpu-sched s | RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 00_warmup_128 | 1.45 | 87.97 | 76.1 | 53.820 | 19.802 | 12.961 | 1.77 | 8.2 | 97.87 |
| 01_Abaseline | 1.33 | 769.69 | 76.6 | 427.149 | 182.428 | 148.549 | 15.17 | 68.8 | 97.87 |
| 03_Cpipe_only | 1.39 | 737.43 | 75.6 | 440.464 | 160.772 | 124.995 | 14.27 | 67.5 | 97.87 |
| 04_Dno_omp | 1.30 | 786.17 | 75.8 | 440.423 | 184.261 | 150.087 | 16.52 | 69.3 | 97.87 |
| 05_Eoptimal | 1.43 | 714.78 | 76.7 | 423.354 | 157.406 | 122.844 | 13.89 | 65.9 | 97.87 |
| 06_Epipe12 | 1.38 | 743.60 | 75.4 | 444.340 | 162.553 | 125.482 | 14.37 | 68.5 | 97.87 |
| 07_Epipe16 | 1.39 | 695.63 | 76.2 | 415.090 | 152.657 | 117.318 | 13.66 | 64.3 | 97.87 |
| 08_ram105 | 1.35 | 760.67 | 74.3 | 460.374 | 162.239 | 126.923 | 15.35 | 68.0 | 92.97 |
| 09_ram112 | 1.32 | 776.09 | 73.7 | 471.648 | 165.365 | 127.814 | 15.17 | 70.3 | 98.79 |
| 10_ram115 | 1.44 | 709.66 | 77.0 | 417.022 | 158.679 | 122.754 | 13.85 | 66.9 | 101.46 |
| 11_ram118 | 1.44 | 709.50 | 77.2 | 415.261 | 159.890 | 123.143 | 13.85 | 67.4 | 104.18 |
| 12_ram120 | 1.50 | 682.46 | 78.3 | 391.895 | 157.877 | 121.501 | 13.89 | 66.9 | 106.42 |
| 13_ram125 | 1.46 | 701.48 | 77.8 | 400.879 | 166.578 | 122.828 | 13.73 | 72.3 | 111.37 |

## methodology

- **Machine:** Apple M1 Ultra, 128 GB unified memory, macOS 26.5.2, internal SSD;
  `iogpu.wired_limit_mb=120832` (118 GB).
- **Model:** GLM-5.2-colibri-int4-perrow — fmt=2 per-row int4, 744B MoE,
  78 layers, 256 experts, topk=8, cache 8 experts/layer.
- **Every run:** `COLI_METAL=1 DIRECT=1 MTP=0`, prompt "Compare the myths of
  Lucifer and Prometheus" (13 tokens), `--ngen 1024` (warmup 128). Prefill is
  ~11.3–11.6 s in all runs and is excluded from tok/s.
- **Expert history frozen:** `colibri_usage.snapshot` (19,178 entries /
  6,373,208 selections, md5 `6ba15f6d1dc0719655232e3144bc50b1`) restored before
  every run → identical pin in all runs: 2478 experts / 46.9 GB mlock'd
  (no compression). Cap auto-raised 8→33 at ram 110 (projected peak 109.9 GB).
- **Single run per config**, warm page cache. The engine is non-deterministic
  run-to-run (see M5 Max report caveats), so treat ~±0.05 tok/s as noise
  (cf. `09_ram112`, slower than ram 110 with no structural cause).
- **`invalid_mtp/` caveat:** it currently holds an aborted 18:00 attempt
  (script invoked with `COLIPATH=./coli` → "No such file or directory",
  wall 0s). The 2026-08-03 MTP-accident logs it was meant to quarantine were
  overwritten by that `mv`; MTP evidence survives only as the numbers in
  `benchmark_run.sh`'s header comment (acceptance 63–74%, 1.63–1.74 tok/fw,
  but 0.07–1.04 tok/s under swap thrash).

## derived metrics

| run | Δ tok/s vs A | disk wait % of decode | misses/token * |
|---|---|---|---|
| 01_Abaseline | — | 55.5% | 140 |
| 03_Cpipe_only | +4.5% | 59.7% | 146 |
| 04_Dno_omp | −2.3% | 56.0% | 145 |
| 05_Eoptimal | +7.5% | 59.2% | 140 |
| 06_Epipe12 | +3.8% | 59.8% | 147 |
| 07_Epipe16 | +4.5% | 59.7% | 143 |
| 08_ram105 | +1.5% | 60.5% | 154 |
| 09_ram112 | −0.8% | 60.8% | 158 |
| 10_ram115 | +8.3% | 58.8% | 138 |
| 11_ram118 | +8.3% | 58.5% | 137 |
| 12_ram120 | +12.8% | 57.4% | 130 |
| 13_ram125 | +9.8% | 57.1% | 133 |

\* misses/token ≈ 599.4 experts-loaded/token × (1 − hit rate).

**Decode budget is fully serial.** At ram 120: 391.9 wait + 157.9 matmul +
121.5 attn + 11.2 other = 682.5 s ≈ the 682.46 s decode wall. Wait and compute
do not overlap, so fixed cost (matmul+attn+other ≈ 290 s) bounds the zero-miss
asymptote at ~3.5 tok/s — unreachable in practice, since each +5 GB of RAM buys
only ~0.5–1pp hit rate.

## conclusions

1. **Config E is best at ram 110** (1.43 tok/s, +7.5% vs baseline A), but the
   M5 Max OMP active-spin trap does **not** reproduce on M1 Ultra: attention GPU
   kernel time is identical between A and D (115.25 vs 115.24 s; on M5 Max it
   was 223→85 s). `COLI_NO_OMP_TUNE` alone is neutral (−2.3%); `PIPE` is the
   only lever that matters (+4.5% alone). Total tuning headroom is small —
   M5 Max gained +79% from the same two levers.
2. **PIPE_WORKERS=8 is the sweet spot.** 12 → 1.38, 16 → 1.39 (vs 1.43 at 8).
   Extra workers don't help once the SSD is saturated.
3. **RAM sweep: soft plateau, no elbow.** Best measured: **1.50 tok/s @
   ram 120**. Hit rate keeps rising (74.3→78.3% over 105→125) and disk wait
   keeps falling (460→392 s), but throughput stops following because
   matmul+attention (~280 s) are fixed costs. No memory-pressure thrash
   anywhere: ram 125 ran at RSS 111.37 GB with the gpu-sched canary flat
   (13.7 s) — the predicted ~109.6 GB thrash zone did not trigger under MTP=0.
   **Operating point: `--ram 120`.**
4. **The bottleneck is expert disk I/O, not RAM.** Disk wait is 55–60% of the
   decode wall in every run (~130–158 misses/token; ≈0.38–0.46 s of every
   token blocked on expert reads). Because wait and compute are serial
   (derived metrics above), each incremental GB of pin buys less than the last.
5. **M5 Max goals not met.** Best 1.50 vs 2.24 tok/s (−33%); the primary target
   ≥2.06 was missed (reached 73%). Two causes: (a) disk wait +76% vs M5 Max at
   the same config (423 vs 241 s @ E/ram 110) — slower storage path;
   (b) expert GPU kernel ~2× slower (66–71 s vs ~34–35 s) despite 80 vs 40 GPU
   cores — the Metal kernels don't scale with core count on the dual-die Ultra.
   Net: M1 Ultra fmt=2 lands between M5 Max's default (1.25) and tuned (2.24)
   configs.

## next steps

1. **MTP re-test only at `--ram 120`–125.** MTP=1 adds ~12–15 GB RSS; at
   ram 110 it swap-thrashed (2026-08-03 accident: 0.07–1.04 tok/s despite
   63–74% acceptance and 1.63–1.74 tok/fw). ram 120 leaves ~17 GB to the
   128 GB wall — one probe (E config + MTP=1 @ ram 120), abort if the
   gpu-sched canary climbs well past its ~14 s baseline.
2. **Bound run-to-run variance:** 2–3 repeats of `05_Eoptimal` and
   `12_ram120` (`FORCE=1` archives old logs to `repeats/`). The `09_ram112`
   1.32 outlier suggests ±5% noise; the E-vs-C and 120-vs-125 rankings need
   confidence intervals.
3. **Measure the SSD directly** with the expert-read pattern (random reads at
   expert granularity) to confirm the disk bottleneck and split the +76%
   disk-wait gap vs M5 Max into hardware vs read-path causes.
4. **Investigate the ~2× expert-kernel gap vs M5 Max** (66–71 s vs ~34–35 s
   kernel): dispatch batch sizes, dual-die buffer placement, fabric overhead.
   If dispatch-bound, larger expert batches per Metal command buffer could
   recover throughput without more RAM.
5. **Write up `METAL-M1ULTRA-FMT2-REPORT.md`** per `docs/metal_fmt2.md`
   Phase 5 and add the `benchmarks.md` community row: Metal on (fmt=2) ·
   `--ram 120` · NO_OMP+PIPE · **1.50 tok/s** · hit 78%.

## environment
# benchmark_run environment snapshot
# date: 2026-08-04T18:02:49Z
# wired_limit_mb: 120832
# usage_lines:    19178
# usage_snapshot: ./colibri_usage.snapshot (6373208 selections)
# ProductName:		macOS
# ProductVersion:		26.5.2
# BuildVersion:		25F84
# model: /Users/joe/mlx-models/GLM-5.2-colibri-int4-perrow
# Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
# /dev/disk3s3   926Gi   700Gi   194Gi    79%    274k  2.0G    0%   /System/Volumes/Data
