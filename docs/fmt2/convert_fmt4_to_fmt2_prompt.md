# Task: Create `c/tools/convert_fmt4_to_fmt2.py` — fmt=4 → fmt=2 converter

Convert the GLM-5.2 fmt=4 (grouped int4, gs=64) container into fmt=2 (per-row int4) so the
Metal backend fully engages (fmt=4 is rejected at every Metal entry point — see
`docs/metal_dispatch_gap.md`). The agreed spec is `docs/metal_fmt2.md` Phase 1.2.

All facts below are already verified — do not re-investigate, just implement.

## Source / destination

- **Source (NFS, read-only):** `/mnt/zfs1/noprot/mlx-lm/models/GLM-5.2-colibri-int4-g64-with-int8-mtp`
  - 141 shards `out-00000.safetensors` … `out-00140.safetensors`
  - `out-mtp-00000.safetensors` (int8 MTP head, 9,959,321,520 bytes)
  - `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `README.md`
  - Skip `_meta/` and `_inflight/` (partial-download leftovers).
- **Dest:** `~/mlx-models/GLM-5.2-colibri-int4-perrow` (553 GiB free, output ~372 GB).

## Container format (verified against real shard headers)

- Every quantized tensor is a **flat 1-D U8** array `name` (packed nibbles, offset binary:
  value = nibble − 8, **low nibble = even element**) plus a **flat 1-D F32** companion `name.qs`.
- Engine detection (`qt_resolve_fmt`, `c/colibri.c:1027`) is purely by byte counts:
  int8 = O·I bytes + O floats; int4 = O·I/2 bytes; `.qs` of O floats → fmt=2,
  O·ceil(I/64) floats → fmt=4.
- Per-tensor classification by element-count ratio nb/ns (weight bytes ÷ scale floats):
  - **32 exactly → fmt=4 g64 (convert)** — shape-independent discriminator
  - int8 (nb = O·I, ns = O) → copy verbatim
  - int4 per-row already (nb = O·I/2, ns = O) → copy verbatim
  - anything else → **abort loudly**
- Verified examples from `out-00000.safetensors`: `o_proj.weight U8 [50331648]` +
  `.qs F32 [1572864]` (6144×16384/2 bytes, 6144×256 groups); experts gate/up/down all
  `U8 [6291456]`; `embed_tokens`/`lm_head U8 [951582720]` + `.qs F32 [154880]` (int8).
- MTP shard tensors are per-row int8/F32 (e.g. `experts.0.gate_proj.weight.qs F32 [2048]` = O)
  → the whole shard is byte-copied, never converted.

## Conversion math (all dims are multiples of 64)

1. Dequantize `w[o,i] = (nib[o,i] − 8) × qs[o×ng + i//64]` to f32, `ng = I/64`.
2. Re-quantize per-row with the **exact** math of `quant_int4` in
   `c/tools/convert_fp8_to_int4.py` (`np.rint`, `absmax/7` clamped ≥ 1e-8, clip −8..7, +8,
   pack low nibble = even element).
3. Write flat U8 `[O*I/2]` + flat F32 `.qs` of **exactly O floats** — the engine then
   auto-detects fmt=2.

## Shape table (O×I) — derive from name + config.json, assert before every conversion

From `config.json`: `hidden_size=6144`, `intermediate_size=12288`,
`moe_intermediate_size=2048`, `n_routed_experts=256`, `n_shared_experts=1`,
`num_hidden_layers=78`, `first_k_dense_replace=3`, `q_lora_rank=2048`, `kv_lora_rank=512`,
`num_attention_heads=64`, `vocab_size=154880`.

| tensor | O×I |
|---|---|
| embed_tokens / lm_head (int8, copy) | 154880×6144 |
| self_attn.q_a_proj | 2048×6144 |
| self_attn.q_b_proj | 16384×2048 |
| self_attn.kv_a_proj_with_mqa | 576×6144 |
| self_attn.kv_b_proj | 28672×512 |
| self_attn.o_proj | 6144×16384 |
| dense MLP layers 0–2 gate/up_proj | 12288×6144 |
| dense MLP layers 0–2 down_proj | 6144×12288 |
| routed + shared experts gate/up_proj | 2048×6144 |
| routed + shared experts down_proj | 6144×2048 |

Assert byte/scale counts match the table before converting each tensor; abort on mismatch.
F32 tensors (norms, `mlp.gate.weight` router, biases) and int8 (embed, lm_head, MTP shard)
are copied.

## Layout

Engine (`c/st.h`) globs any `*.safetensors` and orders by `out-%d`; no index.json.
Keep 1:1 shard mapping with identical filenames; copy the config/tokenizer/README files.

## CLI

- `--indir`, `--outdir` (both required)
- `--dry-run` — header-only scan + classification report: per-format tensor counts,
  validates the shape table against all 142 shard headers, prints size estimates.
- `--selftest` — synthetic fmt=4 tensor → convert → dequant round-trip error small,
  `.qs` has exactly O floats, nibble order verified.
- `--workers N` (default 8) — multiprocessing, mirroring `convert_fp8_to_int4.py`:
  workers convert, the main process writes shards in order → byte-identical output.

## Behavior decisions (already made)

- **Resume:** a re-run skips output shards that already exist with the expected byte size.
- **Preflight:** free-space check on the outdir filesystem (~372 GB needed).
- **MTP shard:** `shutil.copyfile` byte-copy whole.
- Largest tensor is `o_proj` (6144×16384): ~402 MB as f32, ~1.6 GB peak temporaries —
  whole-tensor processing is fine, no row-blocking needed.

## Environment

System python3 has no numpy — `source c/venv.sh` creates `.venv` with numpy+safetensors.
torch is NOT needed (`safetensors.numpy` round-trips U8/F32).

## Do NOT

Do NOT use `coli convert` for this — it only converts FP8→int4 from the HuggingFace repo
and cannot read an int4 container (documented in `docs/metal_fmt2.md` Phase 1.1).

## Verification

1. `source c/venv.sh && python3 c/tools/convert_fmt4_to_fmt2.py --selftest`
2. `python3 c/tools/convert_fmt4_to_fmt2.py --indir /mnt/zfs1/noprot/mlx-lm/models/GLM-5.2-colibri-int4-g64-with-int8-mtp --outdir ~/mlx-models/GLM-5.2-colibri-int4-perrow --dry-run`
3. Full conversion (hours): same command without `--dry-run`.
4. After conversion: `COLI_METAL=1 ./coli run --model ~/mlx-models/GLM-5.2-colibri-int4-perrow --ngen 16 'test'`
   must show non-zero `METAL: blocchi GPU` / `METAL-ATTN:` lines with `fallback CPU 0`.

**Scope for the implementing session: code + tests only** — implement, run `--selftest`
and `--dry-run` against the real source. Do NOT launch the full conversion or the engine run.
