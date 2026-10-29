# Companion: Week 1 Friday, Nsight Compute on your reduction, then the reduction inside vLLM RMSNorm

Two halves:
1. Profile `kernels/w1/reduce.py` with Nsight Compute (`ncu`) and read three numbers: DRAM throughput, achieved occupancy, top warp stall reason.
2. Open vLLM `csrc/layernorm_kernels.cu` and find the reduction inside RMSNorm. Saved copy: `sources/vllm-layernorm_kernels.cu` (snapshot 2026-10-28; upstream line numbers will drift). Its helper `vectorize_read_with_alignment` is in `sources/vllm-vectorization_utils.cuh`.

The definitions come from the saved Modal GPU Glossary pages in `sources/modal-glossary/`. The NVIDIA Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/) is the full reference; you only need the three sections named below.

Source: tutor

## Running ncu on Modal

```
# inside the Modal container, profile one launch of one kernel:
ncu --section SpeedOfLight --section Occupancy --section WarpStateStats \
    -k regex:reduce3 -c 1  python run_kernels.py

# or save everything and open the report on your Mac in the Nsight Compute GUI:
ncu --set full -k regex:reduce -o reduce_report  python run_kernels.py
```

`reduce.py` is a Modal app, so `ncu` must wrap a process **inside** the container. The simplest route: move the body of `run()` into a plain script, and have the Modal function call it with `subprocess.run(["ncu", ...])`. `-k regex:reduce3 -c 1` profiles only the first matching launch; without a filter, ncu replays all 147 launches (7 kernels × 21 launches) many times each, which is slow.

Three risks I have not checked, so verify before Friday:
- The `nvidia/cuda:12.4.1-devel` image may not include `ncu`. If `which ncu` fails, add `apt-get install -y cuda-nsight-compute-12-4` to the image.
- The container may not be allowed to read GPU performance counters. The error is `ERR_NVGPUCTRPERM`. If you see it, check Modal's docs for profiling support before trying anything else.
- Template kernels have mangled names (`_Z7reduce6ILj128EEvPKfPf`); `regex:reduce6` still matches them.

Source: tutor

## Number 1: DRAM throughput

```
Kernels that are memory-bound are limited by the memory bandwidth of the GPU.
Specifically, they are limited by the bandwidth between the GPU RAM and the local cache of the
Streaming Multiprocessors , because the problems of interest for GPU performance generally have
working set sizes much larger than any higher level of the memory hierarchy .
```

Where: the **GPU Speed Of Light Throughput** section (`SpeedOfLight`), row "DRAM Throughput", as a percent of peak. It is the profiler's version of the "% peak" column that `reduce.py` prints.

What to do with it: compare the two. They should be close. If ncu says much higher than your script, your script's timing includes something else (launch gaps). If ncu says much lower, the kernel is reading less from DRAM than you think, for example because data came from L2 (see the L2 warning in `w1-mon-h100-numbers.md`). Below 70% on a memory-bound kernel means the next two numbers tell you why.

Source: artifacts/sources/modal-glossary/memory-bound.txt:5-8

## Number 2: achieved occupancy

```
Occupancy is the ratio of the active warps to the maximum number of active warps on a device.
There are two types of occupancy measurements: Theoretical Occupancy represents the upper limit for
occupancy due to the kernel launch configuration and device capabilities.
Achieved Occupancy measures the actual occupancy during kernel execution, aka on active cycles .
```

Where: the **Occupancy** section, rows "Theoretical Occupancy" and "Achieved Occupancy". The H100 maximum is 64 warps per SM.

Predictions to write down first:
- **Kernels 1-6:** 128 threads = 4 warps per block. One SM takes at most 2048 / 128 = 16 such blocks, which is 64 warps. Shared memory (512 bytes per block) and registers are not limits. Theoretical occupancy 100%; achieved a bit lower because blocks finish and start unevenly.
- **Kernel 7:** `K7_BLOCKS = 132 * 4` = 4 blocks per SM = 16 warps = **25%**. Harris chose few blocks on purpose: each thread adds many elements, so fewer threads are needed.

Occupancy is not a goal in itself (the glossary page says this at lines 30-33). It only matters if there are too few warps to hide memory latency, which is the next number.

Source: artifacts/sources/modal-glossary/occupancy.txt:5-8

## Number 3: top warp stall reason

```
When warps are stalled on accesses to shared memory or on long-running arithmetic instructions, they
are said to be stalled on the "short scoreboard".
When warps are stalled on accesses to GPU RAM, they are said to be stalled on the "long scoreboard".
Both types of stalls are known as scoreboard stalls .
```

Where: the **Warp State Statistics** section (`WarpStateStats`). The CLI prints the largest stall reason in a short note under the section; the GUI shows a bar chart of all reasons.

The names you will see, mapped to the reduction:
- **Stall Long Scoreboard:** waiting for a global memory load (`g_idata[i]`). Expected top reason for a healthy memory-bound kernel: the warps are waiting on HBM, which is the point.
- **Stall Barrier:** waiting at `__syncthreads()`. Expect this to be large in kernels 1-4, which call `__syncthreads()` 8 times per block (once after the load, 7 times in the loop) for only 128 (or 256) elements. Kernels 5-7 call it twice: the last 6 loop steps become warp shuffles.
- **Stall Short Scoreboard:** waiting for shared memory (`sdata`). Expect it higher in kernel 2, which has bank conflicts.

Source: artifacts/sources/modal-glossary/warp-execution-state.txt:31-34

## Little's Law: will kernel 7 have enough warps on an H100?

```
If a GPU has a peak throughput of 1 instruction per cycle and a memory access latency of 400 cycles,
then 400 concurrent memory operations are needed across all active warps in a program.
```

Apply it to HBM: bytes in flight needed = bandwidth × latency. With 3.35 TB/s and an HBM latency of roughly 600-800 ns (my estimate; it is not in the saved sources), that is about **2-2.7 MB in flight** across the GPU, or **15-20 KB per SM**.

Kernel 7 with 4 blocks per SM has 512 threads per SM. Each pass of its `while` loop issues two 4-byte loads per thread: 512 × 8 = **4 KB per SM** in flight, unless the compiler unrolls the loop so that several passes' loads are in flight at once.

Prediction: kernel 7 as written may reach well under 70% of peak on an H100, with **Stall Long Scoreboard** on top and low achieved occupancy. Harris's 2007 GPU had about 86 GB/s, so 4 blocks per SM were enough there. If the profile agrees, try `K7_BLOCKS = 132 * 16` (full occupancy) and compare. This is an estimate; the profile decides.

Source: artifacts/sources/modal-glossary/littles-law.txt:12-13

## vLLM RMSNorm: write your four answers first

```
// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool HasWeight>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
```

RMSNorm, the normalization layer in Llama-style models: for each token's hidden vector `x` of length `hidden_size`, output `y_i = x_i / sqrt(mean(x^2) + eps) × w_i`. One sum of squares per row, then a scale.

Per the README rule, before reading further, write: what it computes, how work splits over blocks and warps, where each tensor lives, and the bound type. My answers are spread through the next sections.

Source: artifacts/sources/vllm-layernorm_kernels.cu:13-17

## Work split: one block per token

```
  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  const int max_block_size =
      batch_invariant_launch ? 1024 : ((num_tokens < 256) ? 1024 : 256);
  dim3 grid(num_tokens);
```

`grid(num_tokens)`: one block per token (row). Each block reduces one row, so there is no cross-block reduction at all, which avoids the problem of Harris slide 4 (blocks cannot wait for each other).

The block size: few tokens → up to 1024 threads per block, so each row is spread over many threads; 256 or more tokens → 256 threads per block, so more blocks fit on each SM at once. The comment says why: SM concurrency, which is the occupancy and latency hiding from the profiler half of today.

Source: artifacts/sources/vllm-layernorm_kernels.cu:253-257

## Vector width: 16 bytes per load

```
          const int calculated_vec_size =
              std::gcd(16 / sizeof(scalar_t), hidden_size);
          const int block_size =
              std::min(hidden_size / calculated_vec_size, max_block_size);
          dim3 block(block_size);
```

`16 / sizeof(scalar_t)` is the number of elements in a 16-byte load: 8 for BF16, 4 for FP32. One 16-byte load per thread is the widest single load instruction, and fewer, wider loads reach peak bandwidth with fewer threads. The `gcd` drops to a smaller width if `hidden_size` is not a multiple of 8.

Example, Llama-3-8B (`hidden_size = 4096`, BF16): vec size 8, so 4096 / 8 = 512 vectors per row; block size 512 threads for fewer than 256 tokens, 256 threads otherwise. With 256 threads, each thread loads 2 vectors.

Source: artifacts/sources/vllm-layernorm_kernels.cu:267-271

## The reduction, part 1: each thread sums its own slice

```
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);
```

Each thread keeps a private `float variance` in a register and adds `x*x` for its elements. Inputs are BF16 but the sum is FP32 (`static_cast<float>`), for the rounding reason from the Wednesday companion's bounds section. The `vec_op` handles 8 elements at a time; `scalar_op` handles leftovers.

This is **Harris kernel 7, "multiple adds per thread"**: do as much as possible sequentially in a register before any thread talks to another.

Source: artifacts/sources/vllm-layernorm_kernels.cu:54-66

## How the slices are assigned: a block-stride loop

```
    for (int i = tid; i < num_vec; i += stride) {
      vin_t tmp = v_in[i];
      vec_op(tmp);
    }
```

`tid` is `threadIdx.x` and `stride` is `blockDim.x`. Thread 0 takes vectors 0, 256, 512, ...; thread 1 takes 1, 257, .... Neighbouring threads read neighbouring 16-byte vectors, so each warp reads 32 × 16 = 512 contiguous bytes: coalesced, as in Harris slide 33. This is the fast path; the rest of the function handles a row whose start address is not 16-byte aligned, with scalar loads for the unaligned head and tail.

Source: artifacts/sources/vllm-vectorization_utils.cuh:149-152

## The reduction, part 2: the block tree is one library call

```
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
```

**This is the reduction you were asked to find.** CUB (NVIDIA's CUDA template library) `BlockReduce` combines one value per thread into one value per block. Inside, it does what Harris kernels 5-6 do: a warp-shuffle reduction inside each warp (no shared memory, no `__syncthreads()`), then one shared-memory step to combine the per-warp results. `TempStorage` is that shared scratch space. The template argument 1024 is the maximum block size; the last argument says how many threads are really there.

So vLLM did not hand-write Harris's 7 versions. It wrote kernel 7's per-thread loop itself and called a library for kernels 5-6's tree. Only thread 0 gets the correct total back.

Source: artifacts/sources/vllm-layernorm_kernels.cu:68-70

## Broadcasting the result to all threads

```
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();
```

Thread 0 turns the sum into the scale factor `1 / sqrt(mean + eps)` (`rsqrtf` is one fast hardware instruction) and writes it to shared memory. `__syncthreads()` makes every thread wait until it is written, then all of them read `s_variance`. This is the standard "reduce, then broadcast through shared memory" pattern; softmax uses it twice (once for the max, once for the sum).

Source: artifacts/sources/vllm-layernorm_kernels.cu:72-75

## The second pass reads the row again

```
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
```

The scaling loop loads `input` a second time instead of keeping it in registers from the first pass. Traffic per element: read x twice, read w, write y. For a 4096-wide BF16 row that is 8 KB, so the second read very likely comes from L1 or L2, not HBM (a guess you can check with the L2 hit rate in `ncu --set full`).

Bound type: about 3 FLOPs per element (square, add, scale) against at least 6 bytes from HBM (read x, read w, write y in BF16). About 0.5 FLOPs per byte, far below the 20 (CUDA core) and 295 (tensor core) ridge points. Memory-bound.

Source: artifacts/sources/vllm-layernorm_kernels.cu:81-83

## Batch invariance: why reduction order is a correctness issue

```
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. In batch-invariant mode the block size must
     not depend on num_tokens, otherwise the same token would use a different
     reduction width (and thus a different floating-point summation order)
     across batches; lock it to 1024 to keep results bit-exact. */
```

Floating-point addition is not associative: `(a + b) + c` can differ from `a + (b + c)` in the last bit. A tree over 256 threads adds in a different order than a tree over 1024 threads. So the same token gives slightly different bits depending on how many other tokens were in the batch. "Batch-invariant mode" fixes the block size so results do not depend on batch size.

Connect this to Tuesday: every one of Harris's 7 kernels adds in a different order, so your `reduce.py` results differ in low bits between kernels. That is why its check uses a relative tolerance of `1e-4`, not equality.

Source: artifacts/sources/vllm-layernorm_kernels.cu:330-336

## Your production-kernel table: first row

```
Kernel:       vLLM rms_norm_kernel
Computes:     y = x / sqrt(mean(x^2) + eps) * w, per token
Bound:        memory (about 0.5 FLOPs per byte)
The trick:    one block per token + 16-byte vector loads + per-thread FP32 sums, then cub::BlockReduce
```

The Week 1 deliverable asks for this table. Compare this row with what you wrote before reading, and mark the differences.

Source: tutor
