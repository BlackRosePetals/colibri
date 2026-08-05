# fmt=2 benchmark summary
generated: 2026-08-05T00:44:26Z

> Everything below the results table is hand-maintained analysis. Re-running
> `benchmark_run.sh` regenerates this file and will overwrite it.
>
> `12_ram120` stopped at **934 tokens** (early EOS): its tok/s is a valid rate,
> its decode seconds are not comparable to the 1024-token runs.
>
> All runs use `--cap 33` flat (script `CAP=`): v1.4.0's fast-SSD default is
> cap=1, which measured 0.98 vs 1.45 tok/s on the warmup. The fmt2.old run
> auto-raised cap 31→38 across the RAM sweep, so the sweep extremes are not
> cap-matched. `08_ram105`: pin auto-capped 46.9→41.9 GB to preserve LRU room
> (v1.4.0 behavior; net-neutral: same 1.35 tok/s as fmt2.old).

| run | tok/s | decode s | hit % | disk wait s | matmul s | attn s | attn gpu-sched s | expert gpu-sched s | RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 00_warmup_128 | 1.45 | 88.15 | 76.4 | 52.976 | 20.769 | 13.029 | 1.80 | 7.8 | 97.87 |
| 01_Abaseline | 1.31 | 781.17 | 76.5 | 431.739 | 188.270 | 149.558 | 15.59 | 64.7 | 97.87 |
| 03_Cpipe_only | 1.40 | 732.20 | 76.2 | 433.144 | 164.210 | 123.725 | 13.82 | 63.6 | 97.87 |
| 04_Dno_omp | 1.32 | 774.95 | 76.8 | 424.816 | 189.181 | 149.360 | 15.68 | 65.0 | 97.87 |
| 05_Eoptimal | 1.40 | 732.75 | 76.1 | 432.530 | 164.748 | 124.398 | 14.73 | 64.3 | 97.87 |
| 06_Epipe12 | 1.41 | 726.80 | 76.6 | 426.491 | 165.480 | 123.708 | 14.15 | 65.1 | 97.87 |
| 07_Epipe16 | 1.37 | 745.03 | 75.9 | 441.514 | 167.586 | 124.903 | 14.17 | 65.7 | 97.87 |
| 08_ram105 | 1.35 | 757.89 | 74.9 | 455.181 | 165.440 | 126.134 | 15.18 | 64.3 | 93.30 |
| 09_ram112 | 1.38 | 740.15 | 76.0 | 439.410 | 165.284 | 124.407 | 14.12 | 64.2 | 98.79 |
| 10_ram115 | 1.37 | 744.92 | 75.8 | 441.157 | 166.637 | 126.009 | 15.30 | 65.2 | 100.15 |
| 11_ram118 | 1.44 | 708.94 | 77.5 | 411.622 | 163.333 | 122.976 | 14.10 | 63.5 | 101.56 |
| 12_ram120 | 1.45 | 645.78 | 77.6 | 375.613 | 149.371 | 110.670 | 12.97 | 57.9 | 102.48 |
| 13_ram125 | 1.50 | 684.55 | 78.7 | 390.247 | 162.081 | 121.086 | 13.78 | 63.5 | 104.83 |

## methodology

- **Machine:** Apple M1 Ultra, 128 GB unified memory, macOS 26.5.2, internal SSD;
  `iogpu.wired_limit_mb=120832` (118 GB).
- **Model:** GLM-5.2-colibri-int4-perrow — fmt=2 per-row int4, 744B MoE,
  78 layers, 256 experts, topk=8, cache 33 experts/layer (`--cap 33`).
- **Engine:** colibri v1.4.0 (post-rebase dev). Pre-rebase v1.2.0 results are
  in `../fmt2.old/` — see "rebase check" below.
- **Every run:** `COLI_METAL=1 DIRECT=1 MTP=0`, prompt "Compare the myths of
  Lucifer and Prometheus" (13 tokens), `--ngen 1024` (warmup 128). Prefill is
  ~11.0–11.8 s in all runs and is excluded from tok/s.
- **Expert history frozen:** `colibri_usage.snapshot` (19,178 entries /
  6,373,208 selections, md5 `6ba15f6d1dc0719655232e3144bc50b1`) restored before
  every run → identical pin in all runs except `08_ram105`: 2478 experts /
  46.9 GB mlock'd (no compression).
- **Single run per config**, warm page cache. The engine is non-deterministic
  run-to-run (see M5 Max report caveats), so treat ~±0.05 tok/s as noise.

## rebase check (v1.2.0 → v1.4.0)

Same machine, same usage snapshot, cap-matched at ram 110 (33 = v1.2.0's
auto-raised value; v1.4.0's fast-SSD default cap=1 measured 0.98 tok/s on the
warmup and was overridden — see `benchmark_run.sh` `CAP=`):

| run | v1.2.0 tok/s | v1.4.0 tok/s | Δ |
|---|---|---|---|
| 00_warmup_128 | 1.45 | 1.45 | 0% |
| 01_Abaseline | 1.33 | 1.31 | −1.5% |
| 03_Cpipe_only | 1.39 | 1.40 | +0.7% |
| 04_Dno_omp | 1.30 | 1.32 | +1.5% |
| 05_Eoptimal | 1.43 | 1.40 | −2.1% |
| 06_Epipe12 | 1.38 | 1.41 | +2.2% |
| 07_Epipe16 | 1.39 (970 tok) | 1.37 | −1.4% |

**Verdict: performance-neutral.** All ram-110 configs land within the ±3.5%
noise band; hit rate, disk wait and attention time are within ~1–2%.
**Watch item:** expert-matmul is +3–5% in the same direction in all five
ram-110 configs (A 182→188, C 161→164, D 184→189, E 157→165 s) — each within
noise, but the consistent sign deserves one repeat before dismissing.

## derived metrics

| run | Δ tok/s vs A | disk wait % of decode | misses/token * |
|---|---|---|---|
| 01_Abaseline | — | 55.3% | 141 |
| 03_Cpipe_only | +6.9% | 59.2% | 143 |
| 04_Dno_omp | +0.8% | 54.8% | 139 |
| 05_Eoptimal | +6.9% | 59.0% | 143 |
| 06_Epipe12 | +7.6% | 58.7% | 140 |
| 07_Epipe16 | +4.6% | 59.3% | 144 |
| 08_ram105 | +3.1% | 60.1% | 150 |
| 09_ram112 | +5.3% | 59.4% | 144 |
| 10_ram115 | +4.6% | 59.2% | 145 |
| 11_ram118 | +9.9% | 58.1% | 135 |
| 12_ram120 | +10.7% | 58.2% | 134 |
| 13_ram125 | +14.5% | 57.0% | 128 |

\* misses/token ≈ 599.4 experts-loaded/token × (1 − hit rate).

**Decode budget is fully serial.** At ram 125: 390.2 wait + 162.1 matmul +
121.1 attn + ~11 other = 684.4 s ≈ the 684.55 s decode wall. Wait and compute
do not overlap, so fixed cost (matmul+attn+other ≈ 294 s) bounds the zero-miss
asymptote at ~3.5 tok/s — unreachable in practice, since each +5 GB of RAM
buys only ~0.5–1pp hit rate.

## conclusions

1. **Config E is best at ram 110** (1.40 tok/s, +6.9% vs baseline A), and the
   M5 Max OMP active-spin trap still does **not** reproduce on M1 Ultra:
   attention GPU time is identical between A and D (149.56 vs 149.36 s);
   `COLI_NO_OMP_TUNE` alone is neutral (+0.8%); `PIPE` is the only lever that
   matters (+6.9% alone). Total tuning headroom remains small.
2. **PIPE_WORKERS=8 is the sweet spot.** 12 → 1.41, 16 → 1.37 (vs 1.40 at 8).
   Extra workers don't help once the SSD is saturated.
3. **RAM sweep: best measured 1.50 tok/s @ ram 125** — ties fmt2.old's best
   (1.50 @ ram 120) at 6.5 GB lower RSS (104.8 vs 111.4 GB; flat cap 33 vs
   auto-raised 38). No memory-pressure thrash anywhere: gpu-sched canaries
   flat at 13.0–15.7 s in every run. The 118/120/125 ranking (1.44/1.45/1.50)
   is within noise, and ram 120's early EOS muddies it further.
   **Operating point: `--ram 120`–125.**
4. **The bottleneck is expert disk I/O, not RAM.** Disk wait is 55–60% of the
   decode wall in every run (~128–150 misses/token). Because wait and compute
   are serial (derived metrics above), each incremental GB of pin buys less
   than the last.
5. **M5 Max goals not met.** Best 1.50 vs 2.24 tok/s (−33%); the rebase did
   not change this — the gap is structural (slower storage path, ~2× expert
   GPU kernel time despite 80 vs 40 cores), unchanged from fmt2.old.

## next steps

1. **Repeats for confidence intervals** (`FORCE=1` archives to `repeats/`):
   2–3× `05_Eoptimal`, `12_ram120`, `13_ram125` — the E-vs-C and 118/120/125
   rankings are within noise, and 12_ram120 needs a full 1024-token run. The
   repeats also settle the **matmul +3–5% watch item** from the rebase check.
2. **MTP re-test at `--ram 120`–125.** Headroom improved post-rebase: cap 33
   projected peak at ram 120 is 114.9 GB (vs 119.2 GB with fmt2.old's cap 36),
   leaving ~13 GB to the 128 GB wall for MTP's +12–15 GB. One probe (E config
   + MTP=1 @ ram 120), abort if the gpu-sched canary climbs well past ~14 s.
3. **Measure the SSD directly** with the expert-read pattern (random reads at
   expert granularity) to split the +76% disk-wait gap vs M5 Max into hardware
   vs read-path causes.
4. **Investigate the ~2× expert-kernel gap vs M5 Max** (66–71 s vs ~34–35 s
   kernel): dispatch batch sizes, dual-die buffer placement, fabric overhead.
5. **Write up `METAL-M1ULTRA-FMT2-REPORT.md`** per `docs/metal_fmt2.md`
   Phase 5 and add the `benchmarks.md` community row: Metal on (fmt=2) ·
   `--ram 125` · NO_OMP+PIPE · **1.50 tok/s** · hit 79%.

## environment
# benchmark_run environment snapshot
# date: 2026-08-04T22:11:34Z
# wired_limit_mb: 120832
# usage_lines:    19178
# usage_snapshot: ./colibri_usage.snapshot (6373208 selections)
# ProductName:		macOS
# ProductVersion:		26.5.2
# BuildVersion:		25F84
# model: /Users/joe/mlx-models/GLM-5.2-colibri-int4-perrow
# Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
# /dev/disk3s3   926Gi   700Gi   194Gi    79%    274k  2.0G    0%   /System/Volumes/Data
