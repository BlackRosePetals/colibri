# DeepSeek V4 target engine (colibri CPU)

[简体中文](deepseek-v4.zh-CN.md)

This is the target-only DeepSeek V4 Flash engine for the first PR of the V4
split. DSpark speculative decoding is intentionally excluded and belongs in a
separate stacked follow-up.

## Scope

- Production code is in `c/deepseek_v4.c`; the experimental public engine and
  session API is in `c/deepseek_v4.h`.
- Official sharded safetensors checkpoints load through shared `st.h`.
- Standard MXFP4 matrix multiplication uses shared `quant.h`.
- Unified `c/coli` routes `run`, `chat`, `serve`, and `web` to V4. Serving keeps
  the engine and caches warm across requests.
- `--no-dspark` is a compatibility no-op. This PR has no DSpark model, memory
  tier, or speculative loop.
- Build targets are x86-64/aarch64 Linux and Windows/MSYS2.

Destroy every session before destroying its engine.

## Shared migration status

| Checkpoint path | Current implementation | Follow-up |
|---|---|---|
| Safetensors index/range reads | shared `st.h` | done |
| fmt7 standard MXFP4 matmul | shared `quant.h` | done |
| fmt7 resident rows16 expert cache | temporary V4-private layout | **TODO:** migrate after upstream exposes a resident rows16 API |
| fmt8 E4M3 + UE8M0 128x128 scales | shared `st_read_scale_f32` + `quant.h` `matmul_fp8` | done |

Only the rows16 resident-cache layout remains V4-private. Its
`TODO(upstream-fmt7-rows16)` marker names the shared API still needed before
that specialized cache layout can be removed.

## Memory policy

A typical checkpoint has 43 transformer layers, hidden size 4096, and 256
routed experts per sparse layer with top-k 6. Dense weights occupy about
6.27 GiB and a resident BF16 output head about 1.06 GiB. Routed-expert weights
are streamed and cached according to the RAM budget.

The planner reserves workspace and a minimum expert working set, then enables
dense/head residency and grows the expert cache when memory permits. Dense
residency is independent of DSpark and works in this target-only build,
including with the legacy `--no-dspark` option.

`--ram GiB` is a planner budget, not an OS-enforced limit. Without it, the
budget is derived from currently available OS memory.

## Download

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

A download can finish with a truncated shard even when the client reports
success. If `st.h` rejects a shard as out of bounds, compare every local shard
size with the Hugging Face repository before treating it as an engine failure.

## Build and use

```bash
cd c
make deepseek-v4
python ./coli run --model /path/to/DeepSeek-V4-Flash --ram 32 \
  "What is the capital of France?"
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli serve --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli web --model /path/to/DeepSeek-V4-Flash --ram 32
```

Generation length: `--ngen` is a ceiling, not a target — answers end at EOS.
If the ceiling exceeds what the context window can hold, the engine clamps it
and says so on stderr; raise `CTX` for genuinely longer answers.

V4 chat uses native model markers. Native serving currently supports greedy
generation and one active KV slot. The HTTP gateway renders OpenAI and
Anthropic tools into V4's native prompt contract, then parses DSML call blocks
back into each protocol; grammar remains unsupported. See the
[per-engine API matrix](api.md#tool-calling-support). The process, weights,
dense tensors, head, and expert cache stay warm across requests, and prefix
checkpoints (next section) let a new request skip the part of its context the
engine has already prefilled — including across sessions and restarts, not
only a strict extension of the previous prompt.

## Prefill: segments, chunks, checkpoints

Prefill is layer-major over token chunks; a serve session that receives an
agent's second turn used to re-prefill the whole conversation because the
window/compressor/indexer state cannot rewind. This section is what changed.

- **Chunks/segments.** Prefill runs over 128-token chunks (`V4_PREFILL_CHUNK`,
  clamp [1,128]; was 64 — one fewer expert sweep per token, +3 % on the CPU
  path) inside 4096-token segments (`V4_PREFILL_SEGMENT`). Segments are
  atomic: the client-cancel poll (`ColiV4SessionAbortFn should_abort` in the
  session options, driven by the gateway when the HTTP client disconnects)
  runs between them, and completed segments are recorded, so an identical
  retry after a client timeout resumes instead of restarting. Every segment
  re-sweeps each layer's routed experts (disk-bound), so fewer/larger segments
  = fewer sweeps at the price of cancel latency. Progress lines
  `v4_prefill N/M tokens` print per segment on multi-segment prompts.
- **Prefix checkpoints** (`V4_PREFIX_CKPT` (1), min length
  `V4_PREFIX_CKPT_MIN` (512)). A snapshot of the attention transaction
  (window KV + compressed slots + compressor/indexer state, the existing
  `ColiV4AttentionSnapshot`) is taken at a shared boundary and restored on a
  later request whose prompt starts with the same bytes. Three capture rules:
  (1) the gateway tells the engine where the rendered system turn ends
  (optional 8th `SUBMIT` header field, `prefix_bytes` in the session options)
  → snapshot there on the very first request; (2) fallback: the longest
  common prefix of two successive fresh prompts; (3) **prompt-end** snapshot
  after every prefill, because agent clients (opencode and friends) re-render
  the assistant reply, so strict "extends everything fed" reuse fails at the
  reply boundary. `V4_PREFIX_CKPT_SLOTS` (4) in-memory slots, LRU with
  prompt-end captures evicted first. Log lines: `v4_ckpt store prefix=N` /
  `prompt_end=N`, `v4_ckpt hit prefix=N`; `V4_PREFIX_LOG=1` explains
  decisions.
- **Persistence.** Prefix captures are written to `<model>/.coli_ckpt/`
  (`V4_PREFIX_CKPT_DISK` (1); `2` also persists prompt-end captures, `0` off;
  ~140 MB for an 8.3k-token prefix, config-fingerprinted, temp+rename) and
  loaded lazily on the first request after a restart, so a fresh serve does
  not re-prefill a known system prompt. Snapshot (de)serialization is the new
  `coli_v4_{attention,compressor,indexer}_snapshot_write/read`.
- **Batched indexer selection** (`V4_IDX_BATCH` (1)): in prefill the Lightning
  Indexer advances per token but scores/selects once per chunk (past
  `index_topk` candidates), with per-token numerics unchanged; `0` restores
  the per-token loop. `V4_IDX_IDENTITY` (0): `1` skips scoring while every
  candidate fits under `index_topk` (index order instead of score order,
  ~8 % faster prefill, changes rounding — off so greedy text matches
  upstream).

Measured with an 8.3k-token agent system prompt (opencode) on the CPU path:
the second and every later turn/session skips the shared prefix entirely
(seconds instead of a full re-prefill), and a serve restart restores it from
disk.

### Environment (added here)

| var | meaning |
|---|---|
| `V4_PREFILL_CHUNK` (128) / `V4_PREFILL_SEGMENT` (4096) | chunk width / atomic segment length |
| `V4_PREFIX_CKPT` (1), `V4_PREFIX_CKPT_MIN` (512), `V4_PREFIX_CKPT_SLOTS` (4), `V4_PREFIX_CKPT_DISK` (1), `V4_PREFIX_LOG` | prefix checkpoints, see above |
| `V4_IDX_BATCH` (1) / `V4_IDX_IDENTITY` (0) | batched indexer selection / identity short-circuit |
| `COLI_V4_ROWS16` (1) | `0` = never repack hot experts into the rows16 layout (reference matvec for every expert; for numerics comparisons) |
| `DSV4_ATTN_PROF`, `DSV4_DECODE_PROF` | per-chunk attention-block / per-token decode profilers on stderr |

## Validation

The tiny safetensors fixture is generated locally, ignored, and not committed:

```bash
python -m pip install -r tools/requirements-deepseek-v4-tiny.txt
make deepseek-v4-tiny-check
```

This covers loading, teacher forcing, greedy decode, long/repeated sessions,
`--no-dspark` compatibility, and two requests through the persistent
`SUBMIT`/`DATA`/`DONE` protocol.

For a real checkpoint:

```bash
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash \
  MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

The oracle is target-only. DSpark on/off speed, acceptance, and token identity
evidence belong to the stacked DSpark PR.

### What "identical output" means on the CPU path

Greedy text is a regression check within one configuration, not an identity
proof across configurations: the hot-expert rows16 kernel and the reference
matvec accumulate in different orders, and the hot set itself depends on the
`.coli_usage` history, chunk width and cache hits, so two runs of *upstream*
can differ after a few dozen tokens. With the variables that make both sides
run the same kernels on the same experts —

```bash
COLI_V4_ROWS16=0 COLI_V4_AUTOPIN=0 COLI_V4_SAVE_USAGE=0 V4_PREFIX_CKPT=0
```

— an 826-token prompt, 48 greedy tokens, `--memory-gb 22`, gives byte-identical
text between upstream `dev` and this branch (2026-08-16). Compare with 48
tokens rather than 8 (short outputs hide the first divergence) and strip the
`TUNE decode` line. The tiny fixture (`make deepseek-v4-tiny-check`) is
token-exact.

## Follow-ups

- Add non-greedy sampling and more serving slots.
- Add shared replacements for the two temporary private quant paths above.
- In the stacked PR, restore DSpark without changing target tokens and report
  DSpark on/off performance and acceptance data.
- Prefix checkpoints are keyed on prompt bytes; a tokenizer-level key would
  also survive whitespace-only differences in the rendered system turn.
