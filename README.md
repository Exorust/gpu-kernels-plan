# GPU Kernels: 6 weeks to GPU MODE

Follow-on to [time-to-first-token](https://github.com/patchy631/time-to-first-token). 5 sessions a week, 30-45 minutes each, 2 buffer days.

**Rule for every kernel:** before you open the file, write down what it computes, how work splits across blocks and warps, where each tensor lives (global, shared, registers), and whether it is memory-bound or compute-bound. Then read the code and mark where you were wrong.

GPU work runs on Modal H100. GPU MODE problems run on GPU MODE's machines through [popcorn-cli](https://github.com/gpu-mode/popcorn-cli).

---

## Week 1: Sharpen the architecture by measuring

You know the architecture. This week turns it into numbers you can predict before you profile.

**Deliverable.** Reduction and online softmax kernels, each at 80% or more of peak memory bandwidth, with a written prediction made before each profile.

| Day | Session |
|---|---|
| Mon | Write H100 numbers from memory (SMs, shared memory per SM, L2, HBM bandwidth, BF16 TFLOPS, roofline ridge point). Check them against the [Modal GPU Glossary](https://modal.com/gpu-glossary). Fix the wrong ones. |
| Tue | *build* · Reduction ([Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), versions 1-7). Predict bandwidth for each version, then measure. |
| Wed | *build* · Fused online softmax ([paper](https://arxiv.org/abs/1805.02867)). Week 4 FlashAttention depends on this. |
| Thu | *profile* · [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) on both kernels: memory workload, warp stall reasons, roofline. |
| Fri | *read* · [Aleksa Gordic, Inside NVIDIA GPUs](https://www.aleksagordic.com/blog/matmul), the PTX and SASS (machine code) sections. Dump the SASS of your softmax with `cuobjdump -sass`. |

## Week 2: vLLM and SGLang kernels

**Deliverable.** One table: kernel, what it computes, bound type, the one trick that makes it fast. Mark kernels both engines share (often FlashInfer).

| Day | Session |
|---|---|
| Mon | vLLM simple kernels: [`layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu), [`activation_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu), [`cache_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu). Map from [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm). |
| Tue | vLLM attention: [`paged_attention.md`](https://github.com/vllm-project/vllm/blob/main/docs/design/paged_attention.md) for the algorithm (the CUDA kernel is gone from `csrc/`), then [`triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py). |
| Wed | vLLM MoE and quantized GEMM: [`topk_softmax_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/moe/topk_softmax_kernels.cu), [`fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py), [Marlin](https://github.com/IST-DASLab/marlin). |
| Thu | SGLang attention (kernels now live in [`python/sglang/kernels/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels)): [`extend_attention.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/extend_attention.py) (prefill on a cached prefix) and [`decode_attention.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/decode_attention.py) (split-KV decode). |
| Fri | SGLang GEMM and MoE: [`aot/csrc/gemm`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/csrc/gemm), [`aot/csrc/moe`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/csrc/moe). Note which are their own and which wrap CUTLASS or DeepGEMM. |

## Week 3: The famous GEMM algorithms

Placed before FlashInfer: FlashInfer's GEMMs are CUTLASS templates, and they only read clearly after you know these techniques.

**Deliverable.** Your own GEMM plotted against cuBLAS, plus a one-page table of named techniques (problem solved, hardware feature needed, repo that uses it).

| Day | Session |
|---|---|
| Mon | *build* · [siboehm CUDA matmul](https://siboehm.com/articles/22/CUDA-MMM), jump to kernels 4-6 (block tiling, vectorized loads). |
| Tue | *build* · Same GEMM in [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Note grouped tile ordering for L2 reuse. |
| Wed | *read* · Hopper: [Colfax TMA](https://research.colfax-intl.com/tutorial-hopper-tma/), [Colfax WGMMA](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/), then [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) (warp specialization, pipelining). |
| Thu | *read* · Work splitting and production: [Stream-K](https://arxiv.org/abs/2301.03598), split-K, persistent kernels; then [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (FP8 with fine-grained scaling). |
| Fri | *read* · Blackwell: [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/). GPU MODE's NVFP4 problems target this hardware. |

## Week 4: FlashInfer and attention algorithms

**Deliverable.** FlashAttention forward in Triton that passes a correctness test against PyTorch, plus a benchmark of FlashInfer's FP8 GEMM and batch decode.

| Day | Session |
|---|---|
| Mon | *read* · [FlashInfer paper](https://arxiv.org/abs/2501.01005), then [`include/flashinfer/gemm/`](https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/gemm): `bmm_fp8.cuh`, `group_gemm.cuh`, one `fp4_gemm_template_sm100.h`. Map each template parameter to a week 3 technique. |
| Tue | *read* · FlashInfer attention: [`include/flashinfer/attention/`](https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/attention) `decode.cuh`, `prefill.cuh`, `scheduler.cuh`. |
| Wed | *read* · [FlashAttention-2](https://arxiv.org/abs/2307.08691) and [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html). Derive the tiled forward pass from your week 1 softmax. Skim [FA-3](https://arxiv.org/abs/2407.08608) for the Hopper changes. |
| Thu | *build* · Triton FlashAttention forward. Use the official tutorial only after yours runs. |
| Fri | *build* · Benchmark FlashInfer FP8 GEMM and batch decode on Modal against `torch.matmul` and vLLM's Triton decode. |

## Weeks 5-6: GPU MODE winning kernels

**Deliverable.** One leaderboard submission and a written comparison against the top three submissions, with the gap explained in roofline terms.

Sources: [reference-kernels problems](https://github.com/gpu-mode/reference-kernels/tree/main/problems) (`nvidia/` NVFP4 set, `amd/` `fp8-mm` `moe` `mla-decode`, `helion/` gated-deltanet), [gpumode.com](https://www.gpumode.com) leaderboards, past submissions in [GPUMODE/kernelbot-data](https://huggingface.co/datasets/GPUMODE/kernelbot-data).

| Day | Week 5: read the winners | Week 6: compete |
|---|---|---|
| Mon | Install popcorn-cli, submit one `pmpp_v2` problem to learn the loop. | Pick a target. Recommended: `nvidia/nvfp4_gemm` or `amd/fp8-mm`. |
| Tue | Top `amd/fp8-mm` submissions. Name each week 3 technique used. | First correct version. |
| Wed | Top `amd/moe` submissions. Compare with vLLM `fused_moe.py`. | Apply the biggest technique from the winners. Profile. |
| Thu | Top `nvidia/nvfp4_gemm` submissions. Compare with FlashInfer `fp4_gemm_template_sm100.h`. | Second iteration. Submit. |
| Fri | Top `amd/mla-decode` submissions. | Write the comparison. |

---

Cut from the 10-week version: PMPP chapters, the transpose exercise, Volkov, siboehm kernels 1-3, CUTLASS 3.x and CuTe docs, FlashAttention-1 and -4, MLA reading, SGLang allreduce, FlashInfer-Bench. Add any back if a week 5 winner uses something you cannot read.
