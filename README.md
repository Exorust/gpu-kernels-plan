# GPU Kernels: 6 weeks to GPU MODE

Follow-on to [time-to-first-token](https://github.com/patchy631/time-to-first-token). 5 sessions a week, 30-45 minutes each, 2 buffer days.

The plan is ordered by kernel family: memory-bound kernels, then GEMM, then attention. Each week learns one family, then reads how vLLM, SGLang, and FlashInfer implement it. Nothing is read before its prerequisites; see [PREREQS.md](PREREQS.md) for the graph.

**Rule for every kernel:** before you open the file, write down what it computes, how work splits across blocks and warps, where each tensor lives (global, shared, registers), and whether it is memory-bound or compute-bound. Then read the code and mark where you were wrong.

GPU work runs on Modal H100. GPU MODE problems run on GPU MODE's machines through [popcorn-cli](https://github.com/gpu-mode/popcorn-cli).

---

## Week 1: Memory-bound kernels

You know the architecture. This week turns it into numbers you can predict before you profile.

One small thing per day. Nothing here needs more than one sitting.

**Deliverable.** A reduction kernel at 70% or more of peak memory bandwidth, a correct online softmax, and one line per kernel: what you predicted, what you measured. Start a table of production kernels (kernel, what it computes, bound type, the one trick that makes it fast); every week adds rows.

| Day | Session |
|---|---|
| Mon | Look up five H100 numbers and write one line on what each one limits: HBM bandwidth, BF16 TFLOPS, shared memory per SM, L2 size, SM count. Source: [Modal GPU Glossary](https://modal.com/gpu-glossary). Compute one more: the roofline ridge point (FLOPs per byte where a kernel stops being memory-bound). |
| Tue | *build* · Reduction, versions 1-3 only from [Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (naive, no divergence, no bank conflicts). Measure bandwidth for each. Skim the slides for versions 4-7; build them only if you have time left. Read with the [slide-by-slide companion](notes/w1-reduction-companion.md). |
| Wed | *paper* · Online softmax ([paper](https://arxiv.org/abs/1805.02867), sections 1-3). Derive the rescale step on paper: why a running max and a running sum are enough for one pass. No code today. Week 4 FlashAttention uses this exact step. |
| Thu | *build* · Online softmax, one row per block. Triton is fine (start from the [fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) and change it to one pass). Check against `torch.softmax`. |
| Fri | *profile + production* · [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) on your reduction. Read three numbers only: DRAM throughput, achieved occupancy, top warp stall reason. Then open vLLM [`layernorm_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu) and find the reduction inside RMSNorm. |
| Buffer | Reduction versions 4-7 if skipped. Dump SASS (machine code) of your reduction with `cuobjdump -sass` and find the shared-memory loads. |

## Week 2: GEMM on CUDA cores and Ampere tensor cores

**Deliverable.** Your own GEMM plotted against cuBLAS: a CUDA-core version and a tensor-core version.

| Day | Session |
|---|---|
| Mon | *build* · [siboehm CUDA matmul](https://siboehm.com/articles/22/CUDA-MMM), kernels 4-6 (block tiling, vectorized loads). |
| Tue | *build* · Same GEMM in [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Note grouped tile ordering for L2 reuse. Week 4 attention and every Triton production kernel depend on this. |
| Wed | *read* · Ampere tensor cores: `mma.sync`, `ldmatrix`, `cp.async`, shared-memory swizzling. [Fast matmul from scratch with tensor cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) (Armbruster). |
| Thu | *build* · Convert your Monday kernel to `mma.sync`. Profile tensor pipe utilization. |
| Fri | *production* · [Marlin](https://github.com/IST-DASLab/marlin) (INT4 weights on `mma.sync`, a quantized GEMM). Then vLLM MoE: [`moe/topk_softmax_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/moe/topk_softmax_kernels.cu) (picks experts, uses your Week 1 softmax) feeding the Triton [`fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) (grouped GEMM). |
| Buffer | vLLM [`activation_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu): a simple memory-bound kernel to add to your table. |

## Week 3: GEMM on Hopper and Blackwell

The densest week. Use both buffer days if needed.

**Deliverable.** A one-page table of named GEMM techniques: problem solved, hardware feature needed, repo that uses it.

| Day | Session |
|---|---|
| Mon | *read* · Hopper: [Colfax TMA](https://research.colfax-intl.com/tutorial-hopper-tma/), [Colfax WGMMA](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/). |
| Tue | *read* · [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog): warp specialization, pipelining, persistent kernels. Then [Stream-K](https://arxiv.org/abs/2301.03598) for wave quantization and split-K. |
| Wed | *read* · Low precision: [NVFP4 post](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) (FP8 E4M3/E5M2, block scaling, micro-scaling). Then [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): FP8 GEMM with fine-grained scaling. |
| Thu | *read* · Blackwell: [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) (tensor memory, 2-SM MMA). GPU MODE's NVFP4 problems target this hardware. |
| Fri | *production* · SGLang GEMM and MoE (kernels live in [`python/sglang/kernels/`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels)): [`aot/csrc/gemm`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/csrc/gemm), [`aot/csrc/moe`](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/csrc/moe). Mark which are their own and which wrap CUTLASS or DeepGEMM. |
| Buffer | [CuTe layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html). Needed in Week 5 if you pick an NVIDIA problem. |

## Week 4: Attention

**Deliverable.** FlashAttention forward in Triton that passes a correctness test against PyTorch, plus a benchmark of FlashInfer batch decode against vLLM's Triton decode.

| Day | Session |
|---|---|
| Mon | *read* · [FlashAttention-2](https://arxiv.org/abs/2307.08691) and [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html). Derive the tiled forward pass from your Week 1 softmax. Skim [FA-3](https://arxiv.org/abs/2407.08608) for what Week 3's Hopper features change. |
| Tue | *build* · Triton FlashAttention forward. Use the official tutorial only after yours runs. |
| Wed | *production* · Paged attention: [`paged_attention.md`](https://github.com/vllm-project/vllm/blob/main/docs/design/paged_attention.md) for the algorithm (the CUDA kernel is gone from `csrc/`), [`cache_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu) (`reshape_and_cache` writes K and V into paged blocks), then vLLM [`triton_unified_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py). |
| Thu | *production* · SGLang [`extend_attention.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/extend_attention.py) (prefill on a cached prefix) and [`decode_attention.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/decode_attention.py) (split-KV decode). |
| Fri | *production* · [FlashInfer paper](https://arxiv.org/abs/2501.01005), then [`include/flashinfer/attention/`](https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/attention) `decode.cuh`, `prefill.cuh`, `scheduler.cuh`. Run the benchmark on Modal. |

## Week 5: Pick one GPU MODE problem, learn what it needs, read its winners

Sources: [reference-kernels problems](https://github.com/gpu-mode/reference-kernels/tree/main/problems), [gpumode.com](https://www.gpumode.com) leaderboards, past submissions in [GPUMODE/kernelbot-data](https://huggingface.co/datasets/GPUMODE/kernelbot-data).

**Mon:** install popcorn-cli, submit one `pmpp_v2` problem to learn the loop, then pick the target. Tue-Wed close the gaps for that target. Thu-Fri read its top submissions.

| Target | Tue-Wed: prerequisites | Thu-Fri: compare winners with |
|---|---|---|
| `nvidia/nvfp4_gemm` (recommended) | [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html), FlashInfer [`fp4_gemm_template_sm100.h`](https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/gemm) | FlashInfer FP4 GEMM, Week 3 tcgen05 |
| `amd/fp8-mm` | [HIP hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html), [AMD matrix cores](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html) (MFMA) | DeepGEMM, Week 2 tensor-core kernel |
| `amd/moe` | Same AMD reading as `fp8-mm` | vLLM `fused_moe.py` |
| `amd/mla-decode` | AMD reading, plus MLA in [DeepSeek-V2](https://arxiv.org/abs/2405.04434) section 2.1 | Week 4 Flash-Decoding, vLLM [`csrc/attention/mla`](https://github.com/vllm-project/vllm/tree/main/csrc/attention/mla) |

Name the technique each winner uses, using your Week 3 table.

## Week 6: Compete

**Deliverable.** One leaderboard submission and a written comparison against the top three, with the gap explained in roofline terms.

| Day | Session |
|---|---|
| Mon | First correct version. Correctness before speed. |
| Tue | Apply the biggest technique from the winners. Profile. |
| Wed | Second iteration. |
| Thu | Submit. Profile the gap to the top three. |
| Fri | Write the comparison. |

---

Left out on purpose: PMPP chapters, Volkov, siboehm kernels 1-3, CUTLASS 3.x design docs, FlashAttention-1 and -4, SGLang allreduce, FlashInfer-Bench. Add any back if a Week 5 winner uses something you cannot read.
