# M1 Ultra Metal Benchmark Plan

## Objective

Benchmark colibri on M1 Ultra (128GB unified memory) using the same methodology as the M5 Max report, with the goal of meeting or exceeding the M5 Max's 2.06 tok/s baseline.

**Critical Context:** Two separate benchmark tracks exist due to the fmt=4 Metal backend gap:
- **fmt=2 (per-row int4):** Full Metal support available — see [`metal_fmt2.md`](metal_fmt2.md)
- **fmt=4 (grouped int4):** Metal backend fixes required — see [`metal_fmt4.md`](metal_fmt4.md)

---

## Hardware Context

- **Machine:** Apple M1 Ultra (2× M1 Max dies, 20+10 CPU cores, 48+32 GPU cores)
- **Memory:** 128GB unified (fabric-connected, ~512GB/s bandwidth per die)
- **SSD:** Internal (typically 7GB/s sequential, ~3-4GB/s random 4K)
- **Target:** ≥2.06 tok/s (M5 Max baseline), ideally 2.2+ tok/s

---

## Track Selection

### fmt=2 Track (Per-Row Int4)

**Status:** Blocked on fmt=2 model conversion (converter script to be written)

**Model:** Converted from the official fmt=4 container with `c/tools/convert_fmt4_to_fmt2.py`
(must be created first — spec in [`metal_fmt2.md`](metal_fmt2.md) Phase 1.2).
`coli convert --group-size 0` does **not** do this: it only converts FP8→int4 from the
HuggingFace repo and cannot read an existing int4 container.

**Advantage:** Full Metal support (no code changes needed)

**Tradeoff:** ~9pp lower quality than fmt=4 (grouped scales)

**Plan:** [`docs/metal_fmt2.md`](metal_fmt2.md)

---

### fmt=4 Track (Grouped Int4)

**Status:** Requires Metal backend implementation

**Model:** Official GLM-5.2-colibri-int4-g64-with-int8-mtp

**Advantage:** Higher quality (official container format)

**Requirement:** Implement fixes A+B+C from [`metal_dispatch_gap.md`](metal_dispatch_gap.md)

**Plan:** [`docs/metal_fmt4.md`](metal_fmt4.md)

---

## Prerequisites (Both Tracks)

### 1. Model Availability

- **fmt=4:** `/mnt/zfs1/noprot/mlx-lm/models/GLM-5.2-colibri-int4-g64-with-int8-mtp` (NFS; no local SSD copy exists)
- **fmt=2:** Convert from fmt=4 using plan in [`metal_fmt2.md`](metal_fmt2.md#phase-1)

### 2. Build with Metal

```bash
make clean && make METAL=1
```

### 3. Verify Metal Backend

```bash
make metal-test
```

All 30+ kernel tests should pass (int8/int4/int2 matmul, MoE block, large-batch GEMM, fused attention).

**Note:** Current tests cover fmt=1/2/3 only. fmt=4 tests added as part of [`metal_fmt4.md`](metal_fmt4.md#phase-4).

---

## Benchmark Methodology (Both Tracks)

### Standard Configurations

Mirror M5 Max Configs A–E from [`METAL-M5MAX-PERF-REPORT.md`](METAL-M5MAX-PERF-REPORT.md):

| Config | Flags | Purpose |
|--------|-------|---------|
| **A** | `COLI_METAL=1 DIRECT=1 MTP=0` | Metal baseline |
| **B** | Default (no extra flags) | Check for OMP spin regression |
| **C** | `+ PIPE=1 PIPE_WORKERS=8` | Add I/O overlap |
| **D** | `+ COLI_NO_OMP_TUNE=1` | Kill OMP active-spin |
| **E** | `D + C` | Combined optimal |

### Workload

- **Prompt:** "Compare the myths of Lucifer and Prometheus"
- **Tokens:** 1024
- **Rationale:** Matches M5 Max methodology for comparability

### Metrics to Record

For each configuration:
- tok/s
- wall time (s)
- expert-disk (s)
- expert-matmul (s)
- attention (s)
- attn GPU kernel (s)
- expert GPU overhead (s)
- expert hit rate (%)
- RSS (GB)

---

## M1 Ultra Specific Tests

### PIPE_WORKERS Scaling

M1 Ultra has 32 CPU cores (vs M5 Max's 18). Test higher worker counts:

```bash
# Beyond PIPE_WORKERS=8
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=12 ./coli run ...
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=16 ./coli run ...
```

### MTP (Multi-Token Prediction)

M5 Max report used MTP=0. Test if MTP=1 helps on M1 Ultra:

```bash
COLI_METAL=1 DIRECT=1 MTP=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 ./coli run ...
```

### Memory Pressure

Find M1 Ultra's compression knee:

```bash
# Test --ram 115, --ram 120
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model ... --ram 115 ...
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model ... --ram 120 ...
```

---

## Documentation

### 6.1 Create Performance Report

**fmt=2 track:** [`METAL-M1ULTRA-FMT2-REPORT.md`](metal_fmt2.md#phase-5)

**fmt=4 track:** [`METAL-M1ULTRA-FMT4-REPORT.md`](metal_fmt4.md#phase-6)

Mirror M5 Max report structure:
- Setup section (hardware, model, workload, flags)
- Results table (configs A-E + M1-specific variants)
- Phase-by-phase analysis
- Recommendations
- Caveats/untested levers

### 6.2 Update benchmarks.md

Add row to community benchmarks table:

```markdown
| Apple M1 Ultra (32 cores) · macOS · 128 GB unified · internal SSD | ~5-6 GB/s cold | Metal on · `--ram 110` · COLI_NO_OMP_TUNE=1 · PIPE=1 | **X.XX tok/s** · hit YY% · [PR #NNN] |
```

**Note:** Create separate rows for fmt=2 and fmt=4 results if both tracks complete.

---

## Success Criteria

### fmt=2 Track

1. **Primary:** Achieve ≥2.06 tok/s (match M5 Max baseline)
2. **Stretch:** Achieve ≥2.24 tok/s (beat M5 Max)
3. **Documentation:** Complete METAL-M1ULTRA-FMT2-REPORT.md
4. **PR:** Submit report + benchmarks.md update

### fmt=4 Track

1. **Primary:** Metal dispatch confirmed (METAL-ATTN and METAL lines present)
2. **Stretch:** Achieve ≥2.06 tok/s (match M5 Max with fmt=4)
3. **Documentation:** Complete METAL-M1ULTRA-FMT4-REPORT.md
4. **PR:** Submit report + benchmarks.md update

---

## Execution Order

### Recommended Sequence

1. **fmt=2 track first** — Immediate benchmark, establishes speed baseline
2. **fmt=4 track second** — After backend fixes, measures quality-correct performance

**Rationale:** fmt=2 requires no code changes and can be benchmarked immediately. fmt=4 requires implementing fixes A+B+C from [`metal_dispatch_gap.md`](metal_dispatch_gap.md).

### Parallel Execution

If resources allow, both tracks can run in parallel:
- fmt=2: Benchmarking while fmt=4: Implementation in progress

---

## References

### Core Documents

- [`metal_dispatch_gap.md`](metal_dispatch_gap.md) — fmt=4 backend gap problem statement
- [`metal_fmt2.md`](metal_fmt2.md) — fmt=2 conversion and benchmark plan
- [`metal_fmt4.md`](metal_fmt4.md) — fmt=4 backend fix implementation plan

### Methodology Reference

- [`METAL-M5MAX-PERF-REPORT.md`](METAL-M5MAX-PERF-REPORT.md) — M5 Max benchmark methodology
- [`README.md`](../README.md) — Dual-SSD, PIPE, COLI_NO_OMP_TUNE tuning guidance

### Related Issues

- **#457** — Metal fmt=4 GEMV support (MERGED)
- **#585** — METAL Attention support for fmt:4 (OPEN)
- **#587** — Metal fmt=4 decode: attention + routed experts (OPEN, needs-rebase)

---

## Next Steps

**Choose your track:**

1. **fmt=2 (immediate):** Follow [`metal_fmt2.md`](metal_fmt2.md) — convert model and benchmark
2. **fmt=4 (backend fix):** Follow [`metal_fmt4.md`](metal_fmt4.md) — implement fixes A+B+C, then benchmark

**Both tracks contribute to the same goal:** Understanding M1 Ultra's performance potential with colibri, with different tradeoffs between speed (fmt=2) and quality (fmt=4).
