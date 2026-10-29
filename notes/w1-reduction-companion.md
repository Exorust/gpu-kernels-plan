# Companion to "Optimizing Parallel Reduction in CUDA" (Mark Harris)

Read this next to the slides: [reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (38 slides). One section per slide. Each section has what the slide says, what it means, and what is different on an H100 today.

**Before you start: three facts about the slides.**
1. They are from 2007 and use the **G80** GPU (GeForce 8800). Its memory bandwidth is 86.4 GB/s. The H100 has 3,350 GB/s, 39 times more. The ideas still hold; the absolute numbers do not.
2. The input is `int`, not `float`. Same size (4 bytes), same reasoning.
3. One trick (slides 21-22, 26, 35) is **unsafe on GPUs since 2017**. The section for slide 22 explains why and gives the safe version.

**Words used throughout**

| Word | Meaning |
|---|---|
| Reduction | Combine many values into one with an operation like `+` or `max`. Here: sum an array. |
| Thread block | A group of threads (here 128) that runs on one SM and can share memory and wait for each other. |
| Shared memory | Fast memory inside each SM, shared by the threads of one block. Declared `__shared__`. |
| `__syncthreads()` | Every thread in the block waits here until all threads in the block arrive. |
| Warp | 32 threads that execute the same instruction at the same time. The hardware schedules warps, not single threads. |
| Divergence | Threads in one warp take different sides of an `if`. The warp runs both sides, one after the other, with some threads switched off. |
| Bank | Shared memory is split into 32 banks. Two threads of one warp reading different addresses in the same bank must wait for each other (a bank conflict). |
| Coalescing | When the 32 threads of a warp read 32 neighbouring addresses, the hardware combines them into a few large memory transactions. |

---

## Slide 1: Title

Nothing to read. Mark Harris works on NVIDIA's developer technology team. He later wrote much of NVIDIA's CUDA blog.

## Slide 2: Parallel Reduction

**Says:** reduction is common, easy to write, hard to make fast. The talk improves one kernel through 7 versions.

**Means:** reduction appears inside most kernels you will read in this plan: the sum in LayerNorm and RMSNorm, the max and sum in softmax, the sum of partial results in split-K GEMM and Flash-Decoding. The 7 versions are a list of the standard GPU performance problems, one per version.

## Slide 3: Parallel Reduction (tree)

**Says:** inside a block, sum as a tree: 8 values become 4, then 2, then 1. Use many blocks so large arrays fit and all SMs stay busy. Question: how do blocks combine their results?

**Means:** the tree takes log2(N) steps. 8 values need 3 steps; 128 values need 7. At each step, half of the remaining values are added to the other half. Each block produces one partial sum. Blocks cannot read each other's shared memory, so the partial sums must go through global memory (HBM).

**Today:** 132 SMs on an H100. You need at least 132 blocks, and usually several per SM, to keep all SMs busy.

## Slide 4: Problem: Global Synchronization

**Says:** there is no way to make all blocks wait for each other inside a kernel. The fix: end the kernel. The next kernel launch starts only after all blocks of the previous one have finished.

**Means:** a kernel launch is a barrier for the whole GPU. Why no barrier inside a kernel? If you launch more blocks than can be resident on the GPU at once, the blocks that are waiting at the barrier hold their SM. The blocks that have not started can never run. Every block waits forever (a deadlock).

**Today:** two other options exist. (a) Cooperative groups `grid.sync()` gives a barrier inside a kernel, but only if every block fits on the GPU at once, which is exactly the limit this slide describes. (b) `atomicAdd` on one global value: each block adds its partial sum to it. Production code (for example NVIDIA's CUB library) uses one or two kernel launches, or atomics.

## Slide 5: Kernel Decomposition

**Says:** level 0 runs 8 blocks and writes 8 partial sums. Level 1 runs 1 block on those 8 values. The same kernel code works for every level.

**Means:** call the kernel again on its own output until one value is left. With 4M elements and 128 threads per block: level 0 has 32,768 blocks, level 1 has 256, level 2 has 2, level 3 has 1.

## Slide 6: What Is Our Optimization Goal?

**Says:** measure GFLOP/s for compute-bound kernels, bandwidth for memory-bound kernels. Reduction does 1 operation per element loaded, so measure bandwidth. G80 peak: 384-bit bus × 1800 MHz effective ÷ 8 = 86.4 GB/s.

**Means:** this is the Monday roofline lesson. 1 add per 4-byte element = 0.25 FLOPs per byte. The H100 CUDA-core ridge point is about 20 FLOPs per byte, so a reduction is far into the memory-bound region. The only number that matters is GB/s as a percentage of peak.

How to compute bandwidth: bytes read ÷ time. For 2^22 ints: 2^22 × 4 = 16.8 MB. Every bandwidth number in the slides is this 16.8 MB divided by the time.

**Today:** peak is 3,350 GB/s on an H100 SXM. No real kernel reaches 100%; a simple copy kernel is the practical ceiling to compare against.

## Slide 7: Reduction #1: Interleaved Addressing (code)

**Says:** the first kernel.

**Means, line by line:**
- `extern __shared__ int sdata[];` shared memory array. Its size is set at launch (the third `<<< >>>` argument).
- `tid = threadIdx.x` is the thread's number inside the block (0-127). `i = blockIdx.x*blockDim.x + threadIdx.x` is its position in the whole array.
- `sdata[tid] = g_idata[i];` each thread copies one element from global memory into shared memory. This read is coalesced: thread 0 reads element 0, thread 1 reads element 1, and so on.
- `__syncthreads();` wait until all 128 values are in shared memory.
- The loop: `s` (the stride) is 1, 2, 4, ..., 64. At each step, threads whose `tid` is a multiple of `2*s` add the value `s` positions to their right.
- Thread 0 writes the block's sum.

(The slide calls the function `reduce0`, slide 9 calls it `reduce1`. Same code.)

## Slide 8: Interleaved Addressing (diagram)

**Says:** 16 values, 4 steps. Step 1 (stride 1): threads 0, 2, 4, ..., 14 work. Step 2 (stride 2): threads 0, 4, 8, 12. Step 3: threads 0, 8. Step 4: thread 0. The sum, 41, ends in position 0.

**Means:** follow one number. Position 0 starts as 10. Step 1: 10 + 1 = 11. Step 2: 11 + 7 = 18. Step 3: 18 + 6 = 24. Step 4: 24 + 17 = 41. Check one step yourself on paper; it makes the next slides easier.

## Slide 9: Reduction #1, the problem

**Says:** the `if (tid % (2*s) == 0)` makes warps highly divergent, and `%` is slow.

**Means:** with 128 threads there are 4 warps (threads 0-31, 32-63, 64-95, 96-127). At step 1 the working threads are 0, 2, 4, ...: half of every warp. All 4 warps must run the add instruction, each with half its threads switched off. At step 2, a quarter of each warp works, but all 4 warps still run. Until the stride reaches 32, every warp runs every step while doing less and less useful work. `%` (integer remainder) takes many instructions on a GPU; division hardware is slow.

## Slide 10: Performance, Kernel 1

**Says:** 2^22 ints (the PDF text shows "222"; it is 2 to the power 22, about 4M) take 8.054 ms, which is 2.083 GB/s. Block size is 128 for every test.

**Means:** 2.083 ÷ 86.4 = 2.4% of peak. The kernel is far from memory-bound. Something other than memory is slow: the instructions.

## Slide 11: Reduction #2 (code)

**Says:** replace `if (tid % (2*s) == 0) sdata[tid] += ...` with `index = 2*s*tid; if (index < blockDim.x) sdata[index] += sdata[index + s];`.

**Means:** the same pairs are added, but different threads do the work. Before: thread 2 handled position 2. Now: thread 1 handles position 2, thread 2 handles position 4. The working threads are now always the lowest-numbered ones. The `%` is gone too.

## Slide 12: Reduction #2 (diagram), new problem: bank conflicts

**Says:** same values, same result, but threads 0-7 do step 1, threads 0-3 do step 2, and so on. New problem: shared memory bank conflicts.

**Means: why divergence is fixed.** At step 1 with 128 threads, threads 0-63 work: warps 0 and 1 are full, warps 2 and 3 have no work and skip the instruction. Whole warps stop instead of running with threads switched off.

**Means: why bank conflicts appear.** Shared memory has 32 banks. A 4-byte value at index `k` is in bank `k mod 32`. At stride 1, thread `t` reads `sdata[2t]`. Thread 0 reads index 0 (bank 0), thread 16 reads index 32 (bank 0 too). Two threads in one warp want different addresses in the same bank, so they wait in turn: a 2-way conflict. At stride 2, thread `t` reads index `4t`; threads 0, 8, 16, 24 all hit bank 0: a 4-way conflict. The conflicts get worse as the stride grows.

## Slide 13: Performance, Kernel 2

**Says:** 3.456 ms, 4.854 GB/s, 2.33 times faster.

**Means:** removing divergence and `%` more than doubled the speed. Still only 5.6% of peak.

## Slide 14: Sequential Addressing (diagram)

**Says:** reverse the order. Step 1 (stride 8): threads 0-7 each add the value 8 positions away. Step 2 (stride 4): threads 0-3. The sum, 41, ends in position 0. "Sequential addressing is conflict free."

**Means:** at each step, thread `t` reads `sdata[t]` and `sdata[t + s]`. Neighbouring threads read neighbouring addresses, so the 32 threads of a warp read 32 different banks. No conflicts. The working threads are still the lowest-numbered ones, so divergence stays fixed.

## Slide 15: Reduction #3 (code)

**Says:** `for (s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }`

**Means:** `s >>= 1` divides `s` by 2. With 128 threads, `s` is 64, 32, 16, 8, 4, 2, 1. This is the version the Tuesday plan stops at.

## Slide 16: Performance, Kernel 3

**Says:** 1.722 ms, 9.741 GB/s, 2.01 times faster. 4.68 times faster than Kernel 1 in total.

**Means:** 11% of peak. Two problems fixed (divergence, bank conflicts) and still far from the memory limit.

## Slide 17: Idle Threads

**Says:** in the first step of the loop, half of the threads do nothing.

**Means:** with 128 threads, step 1 uses 64. Those 128 threads were needed only to load 128 values, and then half of them wait. The waste: a thread was started, given registers, and scheduled to do one load.

## Slide 18: Reduction #4: First Add During Load

**Says:** launch half as many blocks. Each thread loads two values and adds them before writing to shared memory: `sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];`.

**Means:** the first step of the tree now happens while loading, in registers, with no shared memory and no `__syncthreads()`. Each block now covers 256 elements with 128 threads. Note `i + blockDim.x`, not `i + 1`: the second read is also coalesced, because thread 0 reads element 128, thread 1 reads 129, and so on.

## Slide 19: Performance, Kernel 4

**Says:** 0.965 ms, 17.377 GB/s, 1.78 times faster.

**Means:** 20% of peak.

## Slide 20: Instruction Bottleneck

**Says:** at 17 GB/s we are far from the memory limit, and reduction does little math. So the limit is likely instruction overhead: address calculation and loop control. Strategy: unroll loops.

**Means:** for each useful `+`, the loop also computes `tid < s`, `tid + s`, the shift, the loop test, the branch, and `__syncthreads()`. Those extra instructions cost more than the add. Unrolling (writing out the loop steps one by one) removes the loop test and the branch, and lets the compiler compute addresses at compile time.

**Today:** this is the reasoning Nsight Compute automates. On Friday, "warp stall reasons" tells you whether warps wait on memory or on instructions.

## Slide 21: Unrolling the Last Warp

**Says:** when `s <= 32`, only one warp is still working. Its 32 threads execute together, so `__syncthreads()` and `if (tid < s)` are not needed. Unroll the last 6 steps.

**Means:** this was true on G80: a warp's 32 threads always ran in lockstep.

**Today: this assumption is false since the Volta generation (2017, compute capability 7.0).** Volta added independent thread scheduling: threads in one warp can run out of step. Code that relies on implicit lockstep can read values before another thread has written them. See slide 22.

## Slide 22: Reduction #5: Unroll the Last Warp (code)

**Says:** a `warpReduce` function does the last 6 adds (`+32, +16, +8, +4, +2, +1`) with no `__syncthreads()`. The pointer must be `volatile`. The main loop stops at `s > 32`. Note: this saves work in all warps, because without unrolling every warp runs the loop test and the `if` at every step.

**Means:** `volatile` tells the compiler to read and write shared memory on every line, instead of keeping values in registers. Without it, the compiler may keep `sdata[tid]` in a register and never see the other threads' writes.

**Today: do not write this version.** On an H100, use one of these instead:

```cuda
// Option A: shared memory, with warp barriers between the read and the write
if (tid < 32) {
    int v = sdata[tid];
    for (int s = 32; s > 0; s >>= 1) {
        v += sdata[tid + s];
        __syncwarp();          // all reads of this step finish
        sdata[tid] = v;
        __syncwarp();          // all writes of this step finish
    }
}

// Option B (what modern code does): warp shuffle, no shared memory
// Each thread holds its value in a register `v`.
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
// lane 0 now holds the sum of the warp's 32 values
```

`__shfl_down_sync` lets a thread read a register of another thread in the same warp directly, with no shared memory. `0xffffffff` says all 32 threads take part. Option A needs two barriers per step because, in one step, thread 0 reads `sdata[16]` while thread 16 writes it. Option B has no shared memory, so it has no such problem. Prefer B. NVIDIA's blog post [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) explains both.

## Slide 23: Performance, Kernel 5

**Says:** 0.536 ms, 31.289 GB/s, 1.8 times faster.

**Means:** 36% of peak.

## Slide 24: Complete Unrolling

**Says:** if the number of loop steps is known at compile time, the whole loop can be unrolled. Block size is at most 512 and a power of 2, so there are few cases. Use C++ templates to handle any block size.

**Today:** the maximum block size is 1024, not 512.

## Slide 25: Unrolling with Templates

**Says:** make the block size a template parameter: `template <unsigned int blockSize> __global__ void reduce5(...)`.

**Means:** the compiler makes a separate kernel for each block size. Inside each one, `blockSize` is a constant.

## Slide 26: Reduction #6: Completely Unrolled (code)

**Says:** every step is written out, each guarded by `if (blockSize >= 512)`, `>= 256`, and so on. The text in red (the `blockSize` tests) is evaluated at compile time.

**Means:** because `blockSize` is a constant, the compiler deletes the tests that are false and keeps only the needed steps. No loop counter, no loop test, no branch. Only the adds, the `tid < N` tests, and the barriers remain.

**Today:** the `warpReduce` part has the same Volta problem as slide 22. In modern code, `#pragma unroll` on a loop with a compile-time bound gives the same result as manual unrolling.

## Slide 27: Invoking Template Kernels

**Says:** at runtime, a `switch` on the block size picks which compiled version to launch.

**Means:** compile-time constants with a runtime choice. CUTLASS (Week 3) uses the same idea at a much larger scale: one compiled kernel per tile size, picked at runtime.

## Slide 28: Performance, Kernel 6

**Says:** 0.381 ms, 43.996 GB/s, 1.41 times faster.

**Means:** 51% of peak.

## Slide 29: Parallel Reduction Complexity

**Says:** log N steps. Total work N - 1 adds, the same as a sequential loop (work-efficient). With P processors, time is O(N/P + log N).

**Means:** "work-efficient" means the parallel version does not do extra adds compared with a CPU loop. The N/P term is the time to process the data. The log N term is the tree. Inside one block, N = P (one thread per element), so the time is O(log N).

## Slide 30: What About Cost?

**Says:** cost = processors × time. N threads × log N time = N log N, which is more than the N of a sequential loop. Brent's theorem: use N/log N threads, each summing log N values sequentially first. Cost becomes N. This is called algorithm cascading.

**Means:** with one thread per element, most threads are idle for most of the tree (slide 17). If each thread first sums several values in a loop, fewer threads are needed and each one does more useful work before the tree starts.

## Slide 31: Algorithm Cascading

**Says:** each thread loads and sums many elements, then the block does the tree. Brent says about log n elements per thread; in practice more is better: better latency hiding, fewer tree levels, fewer kernel launches with only a few blocks. On G80: 64-256 blocks of 128 threads, 1024-4096 elements per thread.

**Means:** this links to Monday's Little's law. A thread that loops over many elements keeps loads in flight continuously. Fewer blocks also means the second kernel launch (slide 5) has very little work.

**Today:** the H100 equivalent of "64-256 blocks" is a small multiple of 132 SMs. A common choice is a grid-stride loop with 2-4 blocks per SM.

## Slide 32: Reduction #7: Multiple Adds per Thread

**Says:** replace the two-element load with a `while (i < n)` loop that adds two elements per iteration and moves `i` forward by `gridSize = blockSize * 2 * gridDim.x`.

**Means:** this is the grid-stride loop. The whole grid covers `gridSize` elements per iteration, then moves forward together. Each thread accumulates into `sdata[tid]`.

## Slide 33: Reduction #7, the coalescing note

**Says:** same code, with a note: the `gridSize` stride keeps the reads coalesced.

**Means:** the obvious alternative is for each thread to sum a contiguous chunk (thread 0 sums elements 0-1023, thread 1 sums 1024-2047). Then the 32 threads of a warp read addresses 1024 elements apart, and no read can be combined. With the grid stride, in each iteration the threads of a warp read 32 neighbouring elements. Coalesced.

## Slide 34: Performance, Kernel 7

**Says:** 0.268 ms, 62.671 GB/s, 1.42 times faster, 30.04 times faster than Kernel 1. On 32M elements: 73 GB/s.

**Means:** 62.671 ÷ 86.4 = 73% of peak at 4M elements; 73 ÷ 86.4 = 84% at 32M. Larger inputs do better because the fixed costs (kernel launches, the final tree) are a smaller share of the total time.

**Today:** this is the version that should reach the 70% target in the plan. Versions 1-3 alone will not.

## Slide 35: Final Optimized Kernel

**Says:** the complete Kernel 7 on one page.

**Means:** read it top to bottom and name each part: grid-stride loop with two adds (slides 18, 32), unrolled tree with compile-time tests (slide 26), unrolled last warp (slide 22), write of the block sum. For an H100, replace `warpReduce` with the shuffle loop from slide 22.

## Slide 36: Performance Comparison (graph)

**Says:** log-log plot of time against number of elements, from 131,072 (2^17) to 33,554,432 (2^25), one line per kernel.

**Means:** each line rises from left to right because more elements take more time. On the right side, the lines are parallel and evenly spaced: each version is a fixed factor faster at large sizes. On the left side (small inputs), the lines flatten: time stops falling because fixed costs (launch time, the tree) dominate. Kernel 7 is lowest everywhere.

## Slide 37: Types of Optimization

**Says:** the algorithm changes (addressing, cascading) gave 11.84 times combined. The code changes (unrolling) gave 2.54 times combined.

**Means:** fix the algorithm first: what each thread does and where it reads from. Unrolling helps, but only after that.

## Slide 38: Conclusion

**Says:** understand coalescing, divergence, bank conflicts, latency hiding. Use peak numbers to guide the work. Know the complexity theory. Find the type of limit (memory, math, or instruction overhead). Fix the algorithm, then unroll. Use templates.

**Means:** this list is the checklist for every kernel in the plan.

---

## The 7 kernels in one table

| # | Change | Problem it fixes | G80 GB/s | % of 86.4 |
|---|---|---|---|---|
| 1 | Interleaved, `tid % (2s)` | (start) | 2.08 | 2.4% |
| 2 | Strided index | Divergence, slow `%` | 4.85 | 5.6% |
| 3 | Sequential addressing | Bank conflicts | 9.74 | 11% |
| 4 | Add during load | Half the threads idle | 17.38 | 20% |
| 5 | Unroll last warp | Loop overhead (unsafe today; use shuffles) | 31.29 | 36% |
| 6 | Full unroll with templates | Remaining loop overhead | 44.00 | 51% |
| 7 | Many elements per thread | Too many threads, too little work each | 62.67 | 73% |

## What to expect on an H100

The order of the versions should stay the same. The gaps between them will be different, because the H100 has much more bandwidth per SM, faster integer instructions, and better shared memory. I do not know the H100 numbers for each version; measuring them is Tuesday's task. Write your prediction for each version before you run it.
