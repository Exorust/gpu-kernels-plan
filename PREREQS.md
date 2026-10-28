# Prerequisite graph

Each arrow means "must know before". Labels show the week each topic sits in. Every arrow points forward in time: no topic is read before its prerequisites. Dashed nodes in Week 5 are learned only if your chosen GPU MODE problem needs them.

```mermaid
graph TD
  classDef w1 fill:#eef6ff
  classDef w2 fill:#fff7e6
  classDef w3 fill:#eefbea
  classDef w4 fill:#f6efff
  classDef w5 fill:#f0f0f0
  classDef opt stroke-dasharray:5 5,fill:#f0f0f0

  roof["Roofline + H100 numbers (W1)"]:::w1
  red["Reduction (W1)"]:::w1
  smx["Online softmax (W1)"]:::w1
  ncu["Nsight Compute + SASS (W1)"]:::w1
  simple["vLLM norm / activation / cache / topk_softmax (W1)"]:::w1

  tile["siboehm GEMM tiling (W2)"]:::w2
  tri["Triton GEMM (W2)"]:::w2
  mma["Ampere tensor cores: mma.sync, ldmatrix, cp.async (W2)"]:::w2
  marlin["Marlin (W2)"]:::w2
  fmoe["vLLM fused_moe.py (W2)"]:::w2

  hop["Hopper TMA / WGMMA / warp spec (W3)"]:::w3
  sk["Stream-K / split-K / persistent (W3)"]:::w3
  lp["FP8 / NVFP4 formats + scaling (W3)"]:::w3
  dg["DeepGEMM FP8 (W3)"]:::w3
  bw["Blackwell tcgen05 (W3)"]:::w3
  sgemm["SGLang GEMM / MoE (W3)"]:::w3
  cute["CuTe layouts (W3 buffer)"]:::w3

  fa["FlashAttention-2 + Flash-Decoding (W4)"]:::w4
  fabuild["Triton FlashAttention build (W4)"]:::w4
  vattn["vLLM paged + Triton attention (W4)"]:::w4
  sattn["SGLang extend / decode attention (W4)"]:::w4
  fiattn["FlashInfer attention (W4)"]:::w4

  cutedsl["CuTe DSL + FlashInfer FP4 GEMM (W5)"]:::opt
  amd["AMD CDNA: MFMA, LDS, HIP (W5)"]:::opt
  mla["MLA (W5)"]:::opt
  gm_nv["GPU MODE nvidia/nvfp4_gemm"]:::w5
  gm_fp8["GPU MODE amd/fp8-mm"]:::w5
  gm_moe["GPU MODE amd/moe"]:::w5
  gm_mla["GPU MODE amd/mla-decode"]:::w5

  roof --> red & smx & tile
  ncu --> tile
  red --> simple
  smx --> simple & fa
  tile --> tri & mma
  mma --> marlin & hop
  tri --> fmoe & fabuild & vattn & sattn
  simple --> fmoe
  hop --> sk & dg & bw & cute
  lp --> dg & cutedsl
  dg --> sgemm & gm_fp8
  fa --> fabuild & vattn & sattn & fiattn & gm_mla
  vattn --> fiattn
  bw --> cutedsl
  cute --> cutedsl
  cutedsl --> gm_nv
  amd --> gm_fp8 & gm_moe & gm_mla
  fmoe --> gm_moe
  mla --> gm_mla
```

## Changes from the earlier codebase-ordered plan

The earlier plan read vLLM and SGLang in Week 2, before GEMM (Week 3) and attention (Week 4). Six reading tasks came before their prerequisites, and five prerequisites were missing. The current plan fixes both:

- Production code now sits at the end of the week that teaches its algorithm.
- Added: Ampere tensor cores (W2), FP8/NVFP4 formats (W3), CuTe (W3 buffer).
- Moved to Week 5, only for the target that needs them: CuTe DSL and FlashInfer FP4 GEMM (NVIDIA problems), AMD architecture (AMD problems), MLA (`mla-decode`).
