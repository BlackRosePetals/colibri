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
GiB). The default CPU engine expands its BF16 resident text tensors to f32 and
keeps selected FP8 experts in a bounded per-layer cache; 32 GB of RAM is a
practical starting point at the default 8,192-token context. Context state grows
by about 54 KiB per token. There is currently no Qwen3.8 GPU backend.

## What stays on disk

The 51B-parameter PLE table is never materialized in RAM. For each token the
engine hashes its bigram and trigram history, reads sixteen 160-byte FP8 rows,
and applies the checkpoint's scalar PLE scale. Routed experts are likewise read
on demand: the official per-expert E4M3 gate, up, and down matrices are decoded
with their 128 x 128 `weight_scale_inv` blocks into an LRU whose capacity is the
first engine positional argument. The launcher defaults to one expert slot per
layer.

QSA caches the two K/V heads and the indexer's raw key. Complete four-token
blocks are pooled and scored, the best 512 blocks are retained, and a causal
tail of up to three tokens is appended. The main 24-head attention then operates
only on those original tokens. The native model limit is 262,144 tokens;
`Q38_MAXT` defaults to 8,192 and may raise the server limit up to that native
ceiling when the required RAM is available.

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
vector. CI repeats the capacity-one path under ASan and UBSan and verifies that
a config/tensor shape disagreement is refused.

## Supported checkpoint layouts

The released multimodal checkpoint stores text tensors below
`model.language_model`; the standalone upstream text class stores them below
`model`. Both prefixes are accepted. Routed experts may be the official
per-expert block-FP8 matrices or the fused BF16 tensors emitted by the upstream
text class. Other model types and incompatible tensor shapes fail during load.

The weights remain covered by the Qwen Community License 1.0 in the downloaded
checkpoint. They are not redistributed by Colibri.
