# issue_622: [Bug]: Metal prefill GEMM is not token-exact vs CPU on near-tie logits (S >= GEMM_MIN); decode unaffected

- **URL:** https://github.com/JustVugg/colibri/issues/622
- **State:** OPEN (opened 2026-07-25 by lBroth)
- **Labels:** bug, metal

## Summary

On tiny fixture containers, teacher-forced prefill disagrees CPU vs Metal at 2 of 32 positions, deterministically, on stock `dev @ 7b03ce0` (M5 Pro, macOS 26.5.2). Greedy **decode is token-identical** on the same containers. Surfaced during the PR #587 hardware runs; filed separately since it reproduces on stock dev where that PR's code does not exist.

- Reproduces on both a grouped int4 g64 container and an int3-experts/per-row-int4-dense container — **format-agnostic**
- `COLI_METAL_GEMM_MIN=1000` (keep every `matmul_qt` GEMM on CPU; default threshold 16 rows) makes both containers bit-identical to the CPU reference — localizes the culprit to the Metal prefill GEMM at S >= 16 rows
- Explains why decode stays exact: the harness prefills a 12-token prompt, below the 16-row threshold
- Near-tie logits can flip token choice under different FP reduction order — benign for throughput, relevant for bit-exact CPU-vs-GPU validation

## Relevance

- **Goal a (fmt_2_m1u):** When validating the fmt=2 reproduction against CPU output, prefill (S >= GEMM_MIN) may diverge on near-ties — judge on coherence, not bit-identity, or set `COLI_METAL_GEMM_MIN=1000` for an exact A/B.
- **Goal b (fmt_4_m1u):** Fix B modifies `coli_metal_gemm` / `bind_gemv` — the same path this bug lives in. PR #763 (GPU prefill attention) is gated off by default because of this family of divergence.

## See also

- [pr_763](pr_763.md), [pr_587](pr_587.md)
