# Metal Dispatch Gap: fmt=4 Grouped Int4 Not Supported Anywhere in the Metal Backend

> **Scope:** This document is a **problem statement** only. It identifies the fmt=4 Metal backend gap and provides evidence. Implementation plans are in:
> - [`docs/metal_fmt2.md`](metal_fmt2.md) — fmt=2 (per-row int4) conversion and benchmark plan
> - [`docs/metal_fmt4.md`](metal_fmt4.md) — fmt=4 (grouped int4) Metal backend fix plan

---

## Symptom

M1 Ultra (128 GB unified memory) benchmark of GLM-5.2 744B MoE achieves **0.15 tok/s**
despite `COLI_METAL=1` being set. From `c/3_with_pipe_no_omp_tune128ngen.log`:

```
METAL: blocchi GPU 0 | fallback CPU 0 | expert su GPU 0 | setup 0.00s gpu-wall 0.00s (kernel 0.00s) scatter 0.00s
```

Zero GPU blocks dispatched, zero fallback blocks, zero experts on GPU — **zero dispatch
attempts** of any kind.

**Second, equally important signal: there is no `METAL-ATTN:` line in the log at all.**
That line is printed at `colibri.c:4968–4970` only when `aok > 0` (i.e. only when the fused
attention/layer kernel actually ran at least once). Its absence proves the fused attention
path never engaged either. Attention ran entirely on the CPU.

Decode profile from the same log (128 tokens, 877.98s, 127 forwards):

| phase | total | per token | where it ran |
|---|---|---|---|
| expert-matmul | 466.662s | 3.67s | **CPU** (should be GPU) |
| attention | 186.485s | 1.47s | **CPU** (should be GPU) |
| expert-disk wait | 154.126s | 1.21s | CPU/disk (correct) |
| other | 70.708s | 0.56s | CPU |

For comparison, the M5 Max report (`docs/METAL-M5MAX-PERF-REPORT.md`) achieves **2.24 tok/s**
(Config E). **Critical context:** that benchmark was run on **fmt=2 (per-row int4)** weights,
NOT the fmt=4 grouped int4 container used here. The M5 Max report explicitly shows
`fallback CPU 0` and non-zero `attn GPU kernel` times (79s in Config E), proving Metal was
active. Issue #585 documents the exact symptom on fmt=4: **2.39 tok/s on fmt=2 vs 0.81 tok/s
on fmt=4** (3× slowdown) on the same M5 Max hardware.

**Bottom line:** The 2.24 tok/s M5 Max result is valid Metal performance, but only for fmt=2
models. The fmt=4 grouped int4 GLM-5.2 container (gs=64) has **no Metal support anywhere** in
the current backend — this is why M1 Ultra shows zero Metal dispatch despite `COLI_METAL=1`.

## Issue #103 Evidence — How the M5 Max 2.06–2.24 tok/s Benchmark Was Achieved

Direct inspection of the original benchmark report ([issue_103.md](issue_103.md), Issue #103) reveals the M5 Max
result was achieved on a **fmt=2 (per-row int4)** model, NOT the official GLM-5.2 fmt=4 container:

- **Model path**: `/Users/doug/glm52_i4` — a custom/local conversion, not the official
  `GLM-5.2-colibri-int4-g64` container
- **Banner confirms fmt=2**: The log shows `expert@8-bit densa@8-bit` without the
  "cosmetically wrong" mismatch that appears for fmt=4 models (see line 82–88 below)
- **Metal fully engaged**: `fallback CPU 0`, `METAL-ATTN: layer GPU 79794`, and
  `attn GPU kernel 76.36s` prove all three Metal paths were active

**Conclusion**: To reproduce the 2.06–2.24 tok/s results, you must either:
1. Convert the model with **fmt=2 (per-row int4)** settings instead of fmt=4, OR
2. Apply the Metal backend fixes (A+B+C) to support fmt=4 grouped int4

The official GLM-5.2 colibri container uses fmt=4 by design — this is not a conversion
error, but a format choice that currently lacks Metal support.

## Verified: the model container is CORRECT — conversion did NOT fail

This was checked directly against the safetensors headers in
`/Users/joe/mlx-models/GLM-5.2-colibri-int4-g64-with-int8-mtp/` (3551 tensors scanned).

For every quantized tensor, the `.qs` scale count is **exactly `O × ceil(I/64)` floats**,
which is the definition of grouped int4 at gs=64. Examples (layer 0 / layer 77, identical):

| tensor | weight bytes (U8) | `.qs` floats | implies |
|---|---|---|---|
| `self_attn.q_a_proj` | 6,291,456 | 196,608 | O·I=12,582,912 → O·I/64 ✓ fmt=4 g64 |
| `self_attn.q_b_proj` | 16,777,216 | 524,288 | O·I=33,554,432 → /64 ✓ fmt=4 g64 |
| `self_attn.kv_a_proj_with_mqa` | 1,769,472 | 55,296 | O·I=3,538,944 → /64 ✓ fmt=4 g64 |
| `self_attn.kv_b_proj` | 7,340,032 | **229,376** | 28,672 rows × **8 groups** ✓ fmt=4 g64 |
| `self_attn.o_proj` | 50,331,648 | 1,572,864 | O·I=100,663,296 → /64 ✓ fmt=4 g64 |
| `mlp.experts.N.{gate,up,down}_proj` | 6,291,456 | 196,608 | 2048×6144 → /64 ✓ fmt=4 g64 |
| `mlp.shared_experts.*` | 6,291,456 | 196,608 | ✓ fmt=4 g64 |
| `mlp.gate.weight` (router) | F32 [256,6144] | — | f32, not quantized |

Config confirms the shapes: `hidden_size=6144`, `moe_intermediate_size=2048`,
`n_routed_experts=256`, `num_experts_per_tok=8`, `n_shared_experts=1`,
`num_hidden_layers=78`, `first_k_dense_replace=3`, `q_lora_rank=2048`, `kv_lora_rank=512`,
`qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256`, `num_attention_heads=64`.

MTP head: `out-mtp-00000.safetensors` is 9,959,321,520 bytes — matches the sum of the three
int8 sizes the top-level `README.md` documents as correct (3,527,131,672 + 5,366,238,584 +
1,065,950,496 = 9,959,320,752, the ~768-byte delta being one merged safetensors header
instead of three). This is the int8 MTP container, not the broken int4 one.

Plus the strongest evidence of all: **the run produces coherent, well-formed output.** The
CPU `matmul_i4_grouped` path is decoding these weights correctly.

**Conclusion: this is a backend gap, not a bad conversion. Do not re-download or re-convert
the model.**

**Note on the banner as a format diagnostic:** the log line
`== GLM C engine (glm_moe_dsa), cache=8 experts/layer | experts@8-bit densa@8-bit | idot: neon ==`
says "8-bit" for both experts and dense. For fmt=4 models, this is cosmetically wrong:
`qt_from_disk` (`colibri.c:1055`) sees a `name.qs` sidecar and uses the container's
pre-quantized fmt=4 data, ignoring the bits setting. The banner correctly reflects fmt=2
(per-row int4) models (as seen in Issue #103), but misrepresents fmt=4 grouped int4.
This banner can serve as a quick visual check: if it says "8-bit" but you're loading a
grouped-int4 container, the format mismatch is confirmed.

## Root Cause: fmt=4 is rejected at every Metal entry point

GLM-5.2 stores **everything** quantized as `fmt=4` grouped int4 with `gs=64`.
`qt_resolve_fmt` (`colibri.c:1027`) resolves the container: it first classifies the weight
bytes as int4 (`fmt=2`), then `detect_group_size` (`colibri.c:1005`) sees
`ns == O*ng*4` and promotes it to `fmt=4, gs=64` (`colibri.c:1043`). Every path below then
sees `fmt=4`.

### Gate 1 — MoE experts (`moe_submit`, `backend_metal.mm:963`)

```cpp
if (!g_dev || (fmt != 1 && fmt != 2)) return nil;   // <-- rejects fmt=4
```

`mfmt` is set from the first collected expert at `colibri.c:3210` (`if(mfmt<0) mfmt=e->g.fmt;`)
= 4, then passed to `coli_metal_moe_block_begin` (`colibri.c:3242`) →
`moe_submit` (`backend_metal.mm:1079`) → immediate `nil`.

This returns **before** the `g_moe_fb++` fallback counter at `backend_metal.mm:973–978`,
which is exactly why all three counters read zero instead of showing fallbacks:

- `blocchi GPU 0` = `g_moe_ok == 0`
- `fallback CPU 0` = `g_moe_fb == 0`  ← would be non-zero if the gate were passed and `resolve()` failed
- `expert su GPU 0` = `g_moe_experts == 0`

With `mh == NULL`, `colibri.c:3245` leaves `cpu_res=1`, the missed-expert submit at
`colibri.c:3299–3301` hits the same gate, so `metal_done = 0` at `colibri.c:3308` and the
CPU expert loop runs everything.

The `moe_gemv` kernel itself (`backend_metal.mm:70–99`) has no fmt=4 case: it handles
`fmt==2` (per-row int4) and `else` (int8), and applies a single per-row scale *after* the
simd reduction (`yout[row]=acc*sc[o]`, line 98) — structurally incompatible with a scale
that varies along the input dimension.

### Gate 2 — prefill GEMM (`coli_metal_gemm`, `backend_metal.mm:860`)

```cpp
if (!g_dev || (fmt!=1 && fmt!=2)) return 0;
```

plus the caller-side gate in `matmul_qt_ex` (`colibri.c:575`):

```c
if(g_metal_enabled && S>=g_metal_gemm_min && !spec_pinned() && (w->fmt==1||w->fmt==2) && !omp_in_parallel()){
```

So every batched (prefill) projection — `q_a`, `q_b`, `kv_a`, `kv_b`, `o_proj`, and the
dense-layer MLPs of layers 0–2 — skips Metal and runs `matmul_i4_grouped` on the CPU
(`colibri.c:591`).

### Gate 3 — fused decode attention (`colibri.c:2392` and `colibri.c:4231`)

Both fused paths require `l->kv_b.fmt==2`:

```c
&& c->qk_rope==64 && vh==256 && l->kv_b.fmt==2){          // attention_rows,  colibri.c:2392
&& c->qk_nope==192 && c->qk_rope==64 && c->v_head==256 && l->kv_b.fmt==2   // layer_forward, colibri.c:4231
```

`kv_b` is fmt=4 (229,376 scales = 28,672 rows × 8 groups), so **both gates fail on every
layer of every token** → `coli_metal_attn_decode` / `coli_metal_layer_decode` are never
called → no `METAL-ATTN:` line → 186s of decode attention on the CPU.

Note this gate is currently *load-bearing for correctness*, not just performance: the
absorb-core dequant helper is per-row only (`backend_metal.mm:134–135`):

```cpp
inline float a_deqrow(device const uchar* base, int row, int i, device const float* sc){
  device const uchar* w=base+(long)row*((A_KVL+1)/2); uchar b=w[i>>1];
  int val=(i&1)?(b>>4):(b&0xF); return float(val-8)*sc[row]; }   // <-- sc[row], per-row only
```

If the gate were simply relaxed to admit fmt=4 without fixing `a_deqrow`, `a_qabs`
(`backend_metal.mm:136`) and `a_ctx` (`backend_metal.mm:168`) would read `sc[row]` out of a
grouped scale array and produce silently wrong attention. **The gate must be relaxed and
`a_deqrow` made group-aware in the same change.**

### Latent hazard — `bind_gemv` does not validate fmt (`backend_metal.mm:637`)

`bind_gemv` forwards whatever `fmt` it is handed straight into `mm_gemv` without checking it.
`mm_gemv` (`backend_metal.mm:16–64`) handles fmt 1/2/3 and treats **everything else as f32**
(`else` branch, line 56). Today nothing reaches it with fmt=4 because Gate 3 blocks the whole
attention path on `kv_b` alone — but the other five projections' fmts are passed through
*unchecked*. A model with fmt=2 `kv_b` and fmt=4 `o_proj` would pass the gate and then read
packed int4 bytes as f32 floats: garbage output, no error, no fallback. This should be
hardened (`return false` on unsupported fmt) as part of the fix regardless of the fmt=4 work.

## Why fmt=4 Is Special — Scale Layout

- **fmt=1 (int8):** one scale per output row → `scale[o]`.
- **fmt=2 (int4 per-row):** one scale per output row → `scale[o]`.
- **fmt=4 (grouped int4):** `ng = ceil(I / gs)` scales **per output row**, `gs=64` here.

The authoritative layout is the CPU reference `matmul_i4_grouped` (`quant.h:168–202`):

```c
int rb=(I+1)/2; int ng=(I+gs-1)/gs;             // ng is over the INPUT dim I
const uint8_t *w  = q4    + (int64_t)o*rb;
const float   *scl= scale + (int64_t)o*ng;      // per output row
for(int g=0; g*gs<I; g++){ float sc=scl[g]; ... a += (partial dot over group g) * sc; }
```

and confirmed by CUDA (`backend_cuda.cu:195–205`, `absorb_scale` at `:163–168`):

```cuda
const float *scl = scales + (size_t)o * ng;
int g = i / gs; if (g >= ng) g = ng - 1;        // clamp tail of last partial group
sum += xs[i] * weight_at(weights, fmt, row, i) * scl[g];
```

So for a GEMV with input dim `K`: **`ng = ceil(K/gs)`**, and the scale for output row `o`,
input element `i` is **`sc[o*ng + i/gs]`**, where `sc` is already the per-expert/per-tensor
base pointer. Per-gemv `ng` values in this model: gate/up `K=6144 → ng=96`;
down `K=2048 → ng=32`; kv_b absorb `K=512 → ng=8`.

Nibble decode convention: Metal's existing fmt=2 code uses **offset binary**
(`float(int(b&0xF)-8)`, `backend_metal.mm:88`), same as the CPU (`quant.h:195`). fmt=4 uses
the identical nibble encoding — **only the scale lookup changes.** (CUDA looks different
because it pre-XORs weights with `0x88` at upload — `backend_cuda.cu:170`,`:778` — and then
sign-extends; net effect is the same values. Metal needs no such conversion.)

Alignment luck worth exploiting: `detect_group_size` (`colibri.c:1010`) only ever returns
`{16,32,48,64,96,128,192,256}` — **all multiples of 8**. The Metal gemv kernels load 8
weights per iteration (`uchar4` → 8 nibbles), so **an 8-element load never straddles a group
boundary**, and the group scale can be applied once per 8-load instead of per element:
`g = (c*8)/gs`. The strided lane loop (`c += 32`, i.e. 256 elements) also preserves this.

## GitHub Issue Evidence — This Is a Known, Active Gap

Direct inspection of the issue tracker reveals this is not a newly-discovered bug but a
**documented, actively-worked gap** in Metal fmt=4 support. The following issues provide
context, precedent, and partial implementations:

### Active Metal fmt=4 Work

**#585** — "[Feature]: METAL Attention support for fmt:4" (OPEN)
- Documents the exact symptom: `PROF=1 COLI_METAL=1` achieves **2.39 tok/s on fmt=2** vs
  **0.81 tok/s on fmt=4** (3× slowdown) on M5 Max
- Root cause identified: attention path gated to `l->kv_b.fmt==2` only
- Proposed solution matches Fix C in this document

**#587** — "Metal fmt=4 grouped-int4 decode: attention + routed experts" (OPEN, needs-rebase)
- PR implementing the fix for #585
- **Verified performance gain**: 0.81 → **2.13 tok/s** after fixing both attention and
  expert paths on a real GLM-5.2 744B checkpoint
- **Scope limitation**: prefill GEMM path (`coli_metal_gemm`) remains unaddressed —
  this is a decode-only fix
- Test plan includes kernel unit tests vs CPU oracle for both `moe_gemv` and attention paths

**#457** — "metal: grouped-int4 (fmt=4) GEMV support" (MERGED 2026-07-30)
- Fixed the `mm_gemv` kernel for fmt=4 (dense projections, attention `q_a/q_b/kv_a/o`)
- Added proper `gs` parameter threading through `AttnW`, `coli_metal_attn_decode`,
  `coli_metal_layer_decode`, and `bind_gemv`
- **Critical discovery**: traced that `qt_from_disk` allocated fmt=4 scale buffers with
  `falloc` (bare `malloc`, never registered), which **accidentally masked the shader bug**
  — the scale buffer never resolved to GPU, so the broken shader code never executed
- Added `fmt_scale_bytes()` to correctly size fmt=4 scale buffers (`O*ng` floats)
- **Test coverage**: 12 new fmt=4 cases in `metal-test`, including grouped `coli_metal_gemm`
  and fused-decode-path (`run_attn_grouped`) with `qraw` metric to detect uniform-scale bugs

### CUDA Precedent — Same Bugs Hit First

**#298** — "cuda+engine: full fmt=4 support" (MERGED)
- CUDA's complete fmt=4 implementation across all paths
- **1.08 tok/s** on RTX 5070 Ti after fix (vs broken 0.05 tok/s before)
- Established the scale layout convention: `scale[o*ng + g]` with tail clamp `g >= ng → ng-1`
- Added fused `matmul_i4_grouped_pair` for gate+up (expert path optimization)

**#334** — "fmt=4 CUDA attention crash" (CLOSED)
- **Hard crash** (GPU memory fault → system reboot) when attention absorb kernels used
  per-row scales on grouped data
- Root cause: `attention_absorb_kernel` indexed scales as `wscale[row]` instead of
  `wscale[row*ng + k/gs]` for fmt=4
- **Prevention note**: "When adding a new quant format or editing a GPU kernel: any kernel
  that multiplies a quantized weight by a scale must index the scale per-group for fmt=4"

**#464** — "cuda: reject fmt=4 at per-row-only entry points" (MERGED)
- Hardens CUDA to explicitly reject fmt=4 at unsafe entry points (`coli_cuda_matmul`,
  `coli_cuda_expert_mlp`)
- Ensures fallback to CPU rather than silent garbage output
- Precedent for the `bind_gemv` hardening recommended in this document

### Key Insight: Accidental Safety via Allocator Bug

Issue #457's investigation revealed a **critical interaction** that explains why the
system didn't produce garbage output despite the shader bugs:

1. `qt_from_disk` allocated fmt=4 scale buffers with `falloc` (not `qalloc`)
2. `falloc` does not page-align and never calls `coli_metal_register`
3. `resolve()` in `backend_metal.mm` only finds pointers inside registered slabs
4. Therefore `resolve(t->s, ...)` **always fails** for fmt=4 scale buffers
5. `bind_gemv`/`coli_metal_gemm` return `false`/`0` → **CPU fallback**

**Consequence**: The allocator bug kept the broken shader code from ever executing.
Fixing the allocator (as #457 does) without fixing the shader would have **armed the bug**
— scale buffers would resolve, but the shader would read them incorrectly.

This is why #457's commit order matters:
- **First**: `f0fa5a9` — shader fmt=4 branch + host plumbing (capability lands **inert**)
- **Second**: `2ba12d8` — allocator fix (activates the already-correct capability)
- **Third**: `fc0f5a5` — attn-test blind-spot fix (strengthens test coverage)

### Current Status Summary

| Path | Status | Notes |
|---|---|---|
| `mm_gemv` (dense projections) | ✅ Fixed (#457) | Merged 2026-07-30, includes `gs` parameter and fmt=4 branch |
| `moe_gemv` (routed experts) | 🟡 In Progress (#587) | PR open, verified 0.81→2.13 tok/s gain on decode |
| Fused attention (`a_qabs`/`a_ctx`) | 🟡 In Progress (#587) | PR open, requires `a_deqrow` to be group-aware |
| Prefill GEMM (`coli_metal_gemm`) | ❌ Not Addressed | Gate still rejects fmt=4, out of scope of #587 |
| `bind_gemv` return value wiring | ❌ Not Fixed | `encode_attention` ignores return at `:683,684,686,699` |

**Net effect**: Even with #587 merged, the M1 Ultra run in this document would still
experience CPU fallback on prefill (layers 0–2 dense MLPs, batched attention projections).
The 0.15 tok/s → ~0.7–0.9 tok/s improvement requires **all three paths** (A+B+C) to be fixed.

## Impact — Recalibrated, Honest

Per-token decode budget today: 3.67s expert-matmul + 1.47s attention + 1.21s disk wait +
0.56s other ≈ 6.9s → 0.15 tok/s.

- **Fix A only (MoE experts on GPU):** removes ~3.67s/token of CPU matmul, but attention
  (1.47s) stays on CPU → roughly **~0.3 tok/s**. This is why the original MoE-only scope was
  insufficient.
- **Fixes A+B+C (all three Metal paths):** CPU keeps only disk streaming, routing and
  scatter; GPU compute is small (expert gemv is bandwidth-bound: ~11.3 GB/token of expert
  weights at unified-memory speeds ≈ tens of ms) → the wall clock should collapse toward the
  **~1.2s/token disk-wait floor ≈ 0.7–0.9 tok/s**.
- **Beating 2.06–2.24 tok/s additionally requires cutting disk traffic**, not more GPU: the
  run shows 69.2% expert hit rate (pin 7.4% + LRU 61.9%) with RSS 90.96 GB against a 110 GB
  budget on a 128 GB machine. Levers, in `ultra_benchmark_plan.md` Phase 5 order: higher
  `--ram` (115/120) for more LRU/pin residency, more `PIPE_WORKERS` (disk service 3.8s/token
  vs 1.21s/token blocked suggests imperfect I/O parallelism), and — since this box has a ZFS
  pool alongside the internal SSD — a `COLI_MODEL_MIRROR` second copy for aggregate read
  bandwidth (see top-level `README.md`, "Dual-SSD").
- **Fix D (NEON)** does not change the Metal-on path materially; it raises the floor for
  fallbacks and for the dense/indexer matmuls that never go to GPU. Expected speedup:
  3–5× over the scalar implementation, based on `matmul_i4`'s existing NEON body.

**Risk note:** all of the above assumes expert slabs resolve to registered Metal buffers
once the format gate passes. Slabs *are* registered (`coli_metal_register` at
`colibri.c:1364`, `:1581`, `:1609`, `:1795`, `:1809`), but this has never actually executed on
this machine, so `g_moe_fb` (the `resolve()` failure counter) may well be non-zero on the
first successful build. If it is, the next thing to investigate is slab registration
lifetime, not the kernels.

## Implementation Plans

**This document is a problem statement only.** Implementation plans are in:

- [`docs/metal_fmt2.md`](metal_fmt2.md) — fmt=2 (per-row int4) conversion and benchmark plan
- [`docs/metal_fmt4.md`](metal_fmt4.md) — fmt=4 (grouped int4) Metal backend fix plan

**Files involved** (for reference in implementation plans):

**Metal backend — `c/backend_metal.mm`**
- `:16–64` `mm_gemv` kernel — needs fmt=4 branch + `gs`; final `else` must become f32-only
- `:70–99` `moe_gemv` kernel — needs fmt=4 branch + `gs`; `yout=acc*sc[o]` (`:98`) must be skipped for fmt=4
- `:134–135` `a_deqrow` — per-row `sc[row]`, needs grouped lookup
- `:136`,`:168` `a_qabs`, `a_ctx` — callers of `a_deqrow`, need `fmt`/`gs` args
- `:590–593` `coli_metal_matmul` — gate `fmt>3`; also wraps scales as `O` floats (wrong size for fmt=4)
- `:637–651` `bind_gemv` — no fmt validation (latent garbage hazard), needs `gs`
- `:665–701` `encode_attention` — ignores `bind_gemv` return values; needs `gs` plumbing
- `:716`,`:753` `coli_metal_attn_decode`, `coli_metal_layer_decode` — need `gs` params
- `:858–860` `coli_metal_gemm` — gate rejects fmt=4
- `:963` `moe_submit` — the primary gate; `:973–978` fallback counters; `:1000–1007` gemv dispatch lambda
- `:1041`,`:1067` `coli_metal_moe_block`, `coli_metal_moe_block_begin` — need `gs` params

**Metal header — `c/backend_metal.h`**
- `:36` `coli_metal_matmul`, `:69` `coli_metal_layer_decode`, `:85` `coli_metal_gemm`,
  `:106` `coli_metal_attn_decode`, `:138`/`:153` `coli_metal_moe_block`/`_begin` — signatures

**Engine — `c/colibri.c`**
- `:575–577` `matmul_qt_ex` Metal GEMM gate — excludes fmt=4
- `:591` CPU fmt=4 dispatch to `matmul_i4_grouped` (the path everything currently takes)
- `:593` IDOT gate — fmt=4 excluded by design
- `:1005` `detect_group_size` (candidates all multiples of 16), `:1027` `qt_resolve_fmt`, `:1043` fmt=2→4 promotion, `:1067` `t->gs` assignment
- `:2390–2412` `attention_rows` fused-attention gate (`kv_b.fmt==2` at `:2392`)
- `:3203–3234` `MB_BUILD` macro — `mfmt` at `:3210`, needs `mgs` + uniformity validation
- `:3242`,`:3301` MoE submit call sites — need `gs` argument
- `:3308` `metal_done` resolution
- `:4228–4279` `layer_forward` fused-layer gate (`kv_b.fmt==2` at `:4231`)
- `:4966–4974` Metal stats printing (`METAL-ATTN` only when `aok>0` — the missing-line clue)

**CPU kernels — `c/quant.h`**
- `:168–202` `matmul_i4_grouped` — AVX2 only, **no NEON** (Fix D)
- `:145–157` `matmul_i4` NEON body — the template to copy
- `:228–239` `matmul_i4_pair` NEON body — second reference
- `:513` `matmul_i4_grouped_pair` — check whether the fused gate+up grouped variant also lacks NEON (it is used at `colibri.c:279–280` when `S==1`, i.e. on the decode path)

**CUDA reference — `c/backend_cuda.cu`**
- `:101–103` `row_bytes`, `:145–156` `weight_at`, `:163–168` `absorb_scale`,
  `:195–205`/`:219` `quant_matmul`, `:844`/`:868–870` upload-time gs validation

**Tests**
- `c/tests/test_backend_metal.mm` (`:44` matmul, `:90` moe_block, `:163` attn_decode)
- `c/tests/test_gemm_largebatch.mm` (`:62`)
- `c/tests/test_i4_grouped.c` (CPU grouped reference, incl. the fused-pair variant)
- `c/tests/test_grouped_g4_cuda.cu` (CUDA fmt=4 semantics test — useful cross-check)

**Logs / plans**
- `c/3_with_pipe_no_omp_tune128ngen.log` (the analyzed run), `c/baseline.log`, `c/2_with_pipe_no_omp_tune.log`
- `ultra_benchmark_plan.md` (benchmark execution plan; Phase 1 complete but **blocked** on
  fmt=4 Metal support — see this document for the root cause)
