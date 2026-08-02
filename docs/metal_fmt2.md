# Metal fmt=2 Benchmark Plan — M1 Ultra Per-Row Int4

## Objective

Convert the official GLM-5.2 fmt=4 (grouped int4) container to fmt=2 (per-row int4) and benchmark on M1 Ultra to establish a speed baseline with full Metal support.

**Rationale:** The M5 Max 2.24 tok/s record was achieved on fmt=2 weights. This plan measures what M1 Ultra's 80 GPU cores can achieve with the same format.

---

## Prerequisites

- fmt=4 model already present: `~/mlx-models/GLM-5.2-colibri-int4-g64-with-int8-mtp`
- Colibri built with Metal: `make clean && make METAL=1`
- Metal backend verified working: `make metal-test` passes

---

## Phase 1: fmt=2 Conversion

### 1.1 Source Model Format

**Important:** The source model `GLM-5.2-colibri-int4-g64-with-int8-mtp` is **already int4** (fmt=4 grouped, group_size=64), NOT FP8. The pathname `int4-g64` confirms this.

**No existing converter:** There is NO tool to convert fmt=4 → fmt=2 directly. The `coli convert` command only converts FP8→int4.

### 1.2 Required: fmt=4 → fmt=2 Converter

A new converter script must be created to:
1. Read existing packed int4 weights (unchanged between formats)
2. Read grouped scales from `.qs` files
3. Dequantize to approximate FP32
4. Recompute per-row scales
5. Write new `.qs` files with per-row scaling

**Size analysis:**
- Source (fmt=4 grouped): 406 GB
- Target (fmt=2 per-row): ~372 GB
- Savings: ~34 GB (from reduced scale file sizes)

**Workflow (disk-safe):**
```bash
# Option A: Convert in-place using fmt4_to_fmt2.py (recommended)
# 1. Create converter script (see c/tools/convert_fmt4_to_fmt2.py)
# 2. Run conversion from NFS source to SSD destination
python3 c/tools/convert_fmt4_to_fmt2.py \
  --indir /mnt/zfs1/noprot/mlx-lm/models/GLM-5.2-colibri-int4-g64-with-int8-mtp \
  --outdir ~/mlx-models/GLM-5.2-colibri-int4-perrow

# Option B: Delete fmt=4 from SSD first, then convert
rm -rf ~/mlx-models/GLM-5.2-colibri-int4-g64-with-int8-mtp
# Then run converter above
```

**Output:** New directory `~/mlx-models/GLM-5.2-colibri-int4-perrow` (~372 GB)

### 1.2 Verify Conversion

Check the banner on a short run:
```bash
COLI_METAL=1 ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow \
  --ngen 16 'test'
```

**Expected banner:** `expert@8-bit densa@8-bit` (no "cosmetic mismatch" warning)

**Expected Metal lines:**
```
METAL: blocchi GPU N | fallback CPU 0 | expert su GPU E
METAL-ATTN: layer GPU N
```

---

## Phase 2: Baseline Measurement

### 2.1 Config A — Metal Baseline

```bash
COLI_METAL=1 DIRECT=1 MTP=0 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"
```

**Record:**
- tok/s
- wall time (s)
- expert-disk (s)
- expert-matmul (s)
- attention (s)
- attn GPU kernel (s)
- expert GPU overhead (s)
- expert hit rate (%)
- RSS (GB)

### 2.2 Config B — Default (OMP Tuning)

```bash
COLI_METAL=1 DIRECT=1 MTP=0 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"
```

**Check:** If OMP active-spin is present, expect ~39% regression (M5 Max experience)

---

## Phase 3: Tuning Path

### 3.1 Add PIPE (Config C)

```bash
COLI_METAL=1 DIRECT=1 MTP=0 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"
```

### 3.2 Kill OMP Spin (Config D)

```bash
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"
```

### 3.3 Combined Optimal (Config E)

```bash
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"
```

**Target:** ≥2.24 tok/s (match or beat M5 Max by leveraging M1 Ultra's 80 GPU cores)

---

## Phase 4: M1 Ultra Specific Tests

### 4.1 Vary PIPE_WORKERS

```bash
# Test higher worker counts for M1 Ultra's 32 CPU cores
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=12 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"

COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=16 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"
```

### 4.2 Test MTP

```bash
# M5 Max had MTP=0; test if MTP=1 helps on M1 Ultra
COLI_METAL=1 DIRECT=1 MTP=1 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 110 \
  "Compare the myths of Lucifer and Prometheus"
```

### 4.3 Memory Pressure Test

```bash
# Test --ram 115, --ram 120 to find M1 Ultra's compression knee
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 115 \
  "Compare the myths of Lucifer and Prometheus"

COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ram 120 \
  "Compare the myths of Lucifer and Prometheus"
```

---

## Phase 5: Documentation

### 5.1 Create METAL-M1ULTRA-FMT2-REPORT.md

Mirror the M5 Max report structure:
- Setup section (hardware, model, workload, flags)
- Results table (configs A-E + M1-specific variants)
- Phase-by-phase analysis
- Comparison with M5 Max fmt=2 results
- Recommendations
- Caveats/untested levers

### 5.2 Update benchmarks.md

Add row to community benchmarks table:
```markdown
| Apple M1 Ultra (32 cores) · macOS · 128 GB unified · internal SSD | ~5-6 GB/s cold | Metal on (fmt=2) · `--ram 110` · COLI_NO_OMP_TUNE=1 · PIPE=1 | **X.XX tok/s** · hit YY% · [PR #NNN] |
```

---

## Success Criteria

1. **Primary:** Achieve ≥2.06 tok/s (match M5 Max baseline)
2. **Stretch:** Achieve ≥2.24 tok/s (beat M5 Max)
3. **Documentation:** Complete METAL-M1ULTRA-FMT2-REPORT.md with full methodology
4. **PR:** Submit report + benchmarks.md update

---

## Notes

- **Workload:** Use identical prompt ("Compare the myths of Lucifer and Prometheus") and 1024 tokens
- **Consistency:** Run each config multiple times, report best/warm-cache numbers
- **RSS:** Track RSS throughout, note when compression kicks in
- **Expert hit rate:** Record hit rate for each config
- **Thermals:** Monitor if M1 Ultra throttles during long runs (1024 tokens @ ~2 tok/s = ~8-10 min)
- **Quality caveat:** fmt=2 (per-row) has ~9pp lower quality than fmt=4 (grouped scales) — this is expected and documented in quant_ablation.py
- **Source format:** The source model is fmt=4 (grouped int4, group_size=64), NOT FP8. Conversion requires a fmt=4→fmt=2 converter, not `coli convert`.
- **Disk space:** fmt=4 source is 406 GB; fmt=2 output is ~372 GB. Delete fmt=4 from SSD before conversion to avoid 778 GB peak requirement.

---

## References

- [M5 Max Performance Report](METAL-M5MAX-PERF-REPORT.md)
- [Metal Dispatch Gap](metal_dispatch_gap.md) — fmt=4 backend gap context
- [ultra_benchmark_plan.md](ultra_benchmark_plan.md) — overall M1 Ultra benchmark plan
