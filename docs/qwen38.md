# Qwen3.8-Flash-Next on colibri

`c/qwen38.c` runs the language model in
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
directly from the official safetensors shards. No conversion or second copy of
the weights is required. This contribution is deliberately **text-only**:
Colibri does not load or advertise the checkpoint's vision encoder, and it does
not use the optional MTP layer.

The upstream language model has 125B ordinary parameters with 6B activated,
plus a 51B hashed n-gram embedding. It has 48 layers arranged as 12 repetitions
of three Gated DeltaNet layers and one Qwen Sparse Attention layer. Every layer
uses a four-branch gated residual and a 512-expert top-10 MoE plus one shared
expert.

## Download and run

Pin the checkpoint revision so a later upstream update cannot silently change
the local tensor contract:

```sh
hf download Qwen/Qwen3.8-Flash-Next-FP8 \
  --revision bcd9f01ddc9cff2316eb84281bebcd5b058bddce \
  --local-dir ~/Models/Qwen3.8-Flash-Next-FP8

make -C c qwen38
COLI_MODEL=~/Models/Qwen3.8-Flash-Next-FP8 ./c/coli chat
```

`coli serve` and `coli web` use the same text-only gateway path. Qwen3.8 thinks
by default; `reasoning_effort` accepts `low`, `medium`, `high`, and `xhigh`, and
`enable_thinking: false` emits the model's official empty thinking prefix.
Tools, image content, audio, and grammar constraints are rejected explicitly.

The official FP8 repository is about 185.5 GB in decimal units (roughly 173
GiB). The default CPU engine keeps resident BF16 matrices in their native
two-byte representation and selected experts in native block-FP8. Activations
are FP32; BF16 matrices use FP32 accumulation, while native FP8 uses FP32 dot
products within each 128-column block and FP64 accumulation across the scaled
blocks. The bounded per-layer cache therefore spends about one quarter of the
previous memory per FP8 expert. Context state grows by about 54 KiB per token.
There is currently no Qwen3.8 GPU backend.

## What stays on disk

The 51B-parameter PLE table is never materialized in RAM. For each token the
engine hashes its bigram and trigram history, reads sixteen 160-byte FP8 rows,
and applies the checkpoint's scalar PLE scale. Routed experts are likewise read
on demand: the official per-expert E4M3 gate, up, and down bytes and their
128 x 128 floating-point `weight_scale_inv` blocks remain native in an LRU whose
capacity is the first engine positional argument. The canonical FP8 matmul
decodes values during accumulation rather than materializing three FP32 matrices
per slot. The launcher defaults to one expert slot per layer.

Prompt execution is expert-major in bounded chunks: it routes up to 32 rows,
groups their assignments by expert, and consumes cache-sized parallel load
groups when the complete demand set is larger than the configured LRU. Shared
expert and DeltaNet projections are batched over the same bounded window;
DeltaNet convolution and recurrent updates remain token-causal. The private
chunk workspace is capped at 64 MiB independently of prompt length.

The serve path owns one hybrid prefix slot. When the next prompt begins with
the exact cached token sequence, QSA K/V/index rows are reused in place and the
saved DeltaNet/PLE recurrent state is restored before evaluating only the
extension. An identical prompt also reuses its saved final logits. Mismatched
or shorter prompts reset every hybrid component rather than guessing at cache
identity. Decode and cancellation mutate only live state, not the published
prompt snapshot.

QSA caches the two K/V heads and the indexer's raw key. Complete four-token
blocks are pooled and scored, the best 512 blocks are retained, and a causal
tail of up to three tokens is appended. The main 24-head attention then operates
only on those original tokens. The native model limit is 262,144 tokens;
`Q38_MAXT` defaults to 8,192 and may raise the server limit up to that native
ceiling when the required RAM is available.

## Performance telemetry

Every served request reports its own routed-expert cache hit rate; persistent
engine counters are differenced at request boundaries rather than exposed as a
cumulative percentage. `DONE` decode throughput is based on completed
inter-token intervals: the first token comes from prefill, so a one-token reply
correctly reports zero decode tok/s instead of an artificial near-infinite rate.
The request wall time, prompt/completion counts, expert read/wait/matmul time,
sequence-mixer time, LM-head time and actual forward count are emitted through
the shared `PROF` frame consumed by `tools/datapoint.py`.

Set `COLI_TIMERS=1` for the finer Qwen-specific breakdown on stderr: expert
reads, FP8 expansion, routed and shared experts, resident matmuls, DeltaNet, QSA
indexing, QSA attention, PLE and the LM head. Architecture-phase times overlap
the resident-matmul counter by design; expert disk service also overlaps the
synchronous miss-wait value in `PROF`.

## Correctness gate

The tiny oracle is generated from the `Qwen4ExpForCausalLM` class in
`transformers==5.16.1`, the first release that exports the Qwen4-Exp text
class:

```sh
python -m pip install -r c/tools/requirements-qwen38-tiny.txt
make -C c qwen38-tiny-check
```

It exercises Gated DeltaNet, sparse QSA above its token budget, PLE, four gated
residual streams, routed and shared experts, prefill, cached decode, and LRU
eviction. The gate checks both greedy token IDs and the final upstream logit
vector, and runs both native-BF16 and expanded-FP32 resident modes at cache
capacities one and four. CI repeats the capacity-one path under ASan and UBSan
and verifies that a config/tensor shape disagreement is refused.

## Supported checkpoint layouts

The released multimodal checkpoint stores text tensors below
`model.language_model`; the standalone upstream text class stores them below
`model`. Both prefixes are accepted. Routed experts may be the official
per-expert block-FP8 matrices or the fused BF16 tensors emitted by the upstream
text class. `Q38_NATIVE_FP8=0` restores the former expanded-FP32 expert cache for
numerical/performance A/Bs; `Q38_NATIVE_BF16=0` does the same for resident and
routed BF16 matrices. Other model types, unsupported scale encodings, and
incompatible tensor shapes fail during load.

`Q38_PREFILL_BATCH=0` restores row-at-a-time prompt execution for controlled
A/B diagnosis. It does not change the single-token decode path.

The weights remain covered by the Qwen Community License 1.0 in the downloaded
checkpoint. They are not redistributed by Colibri.
