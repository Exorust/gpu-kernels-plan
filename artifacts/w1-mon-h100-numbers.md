# Companion: Week 1 Monday, five H100 numbers and the ridge point

The task (README.md line 23): look up five H100 numbers, write one line on what each one limits, then compute the roofline ridge point. The Modal GPU Glossary pages are saved as text in `sources/modal-glossary/` (snapshot 2026-10-28), so the quotes below point at those files.

How to use this: read each glossary page first, then the matching section here. Write your own one-line "what it limits" before you read mine.

Source: tutor

## The task as written

```
| Mon | Look up five H100 numbers and write one line on what each one limits: HBM bandwidth, BF16 TFLOPS, shared memory per SM, L2 size, SM count. Source: [Modal GPU Glossary](https://modal.com/gpu-glossary). Compute one more: the roofline ridge point (FLOPs per byte where a kernel stops being memory-bound). |
```

The point of the day is not the numbers. It is the habit of predicting a kernel's speed from the numbers before you measure it. Every number below ends with a prediction for `kernels/w1/reduce.py`, which you build on Tuesday.

Source: README.md:23-23

## Background: the roofline has two roofs

```
The "compute roof" is a horizontal line with height equal to the arithmetic bandwidth .
The "memory roof" is a slanted line with slope equal to the memory bandwidth .
```

Arithmetic intensity is FLOPs done per byte moved to or from GPU RAM (HBM). Plot a kernel at its arithmetic intensity on the x-axis. If it sits left of where the roofs meet, the slanted memory roof is lower, so memory bandwidth sets its best possible speed. If it sits right, the flat compute roof sets it.

Source: artifacts/sources/modal-glossary/roofline-model.txt:13-14

## Numbers 1 and 2: HBM bandwidth 3.35 TB/s, BF16 989 TFLOP/s

```
System (Compute / Memory) Arithmetic Bandwidth (TFLOPs/s) Memory Bandwidth (TB/s) Ridge Point
(FLOPs/byte) A100 80GB SXM BF16 TC / HBM2e 312 2 156 H100 SXM BF16 TC / HBM3 989 3.35 295 B200 BF16
TC / HBM3e 2250 8 281 H100 SXM FP8 TC / HBM3 1979 3.35 592 B200 FP8 TC / HBM3e 4500 8 562 B200 FP4
TC / HBM3e 9000 8 1125
```

The scraper flattened a table. Read the H100 row as: **H100 SXM, BF16 on tensor cores (TC), HBM3: 989 TFLOP/s, 3.35 TB/s, ridge point 295 FLOPs/byte.** The FP8 row doubles the FLOPs (1979) with the same memory, so its ridge point doubles to 592.

- **HBM bandwidth (3.35 TB/s) limits every memory-bound kernel.** Lower bound on its time = bytes moved / 3.35e12. Reductions, softmax, RMSNorm, and decode attention all live here.
- **BF16 tensor-core FLOPs (989 TFLOP/s) limits compute-bound kernels.** Lower bound = FLOPs / 989e12. Only matrix multiplies (GEMM) and prefill attention get here. 989 is the dense number; NVIDIA's datasheet also prints a "with sparsity" number that is twice as large. Ignore that one.

Source: artifacts/sources/modal-glossary/arithmetic-intensity.txt:43-46

## Prediction for reduce.py from number 1

```
BLOCK = 128          # threads per block, as in the slides
N = 1 << 26          # 67M floats = 268 MB
K7_BLOCKS = 132 * 4  # kernel 7 grid: 4 blocks per H100 SM (slide 31)
```

The reduction reads 2^26 floats × 4 bytes = 268 MB once. Best possible time = 268e6 / 3.35e12 = **80 µs**. The Week 1 deliverable asks for 70% of peak = 2345 GB/s = **114 µs**. Write these two numbers down now; on Tuesday the script prints ms and GB/s for each kernel and you compare.

Risk to check on Tuesday: these numbers assume an H100 **SXM** (3.35 TB/s, 132 SMs). An H100 PCIe has about 2.0 TB/s and 114 SMs. If Modal gives you a PCIe card, the 3350 in `reduce.py` and `K7_BLOCKS` are wrong. Print `cp.cuda.Device().attributes["MultiProcessorCount"]` once to confirm 132.

Source: kernels/w1/reduce.py:20-22

## Number 3: shared memory per SM, 228 KB

```
Consider an NVIDIA H100 GPU, which has these specifications: Maximum warps/SM: 64 Maximum blocks/SM:
32 (32 bit) Registers: 65536 Shared memory (smem): 228 KB For a kernel using 32 threads per thread
block , 8 registers per thread , and 12 KB shared memory per thread block , we end up limited by
shared memory : 64 > 1 = warps/block = 32 threads/block ÷ 32 threads/warp 32 < 256 =
blocks/register-file = 65,536 registers/register-file ÷ (32 threads/block × 8 registers/thread) 32 =
blocks/SM 19 = blocks/smem = 228 KB/smem ÷ 12 KB/block Even though our register file is big enough
to support 256 thread blocks concurrently, our shared memory is not, and so we can only run 19
thread blocks per SM , corresponding to 19 warps .
```

**Shared memory limits how many blocks fit on one SM at once, and how big a tile a block can hold.** The example: 12 KB per block → only 19 blocks per SM.

For `reduce.py` it is not a limit: each block uses `BLOCK * 4` = 512 bytes. It becomes the main limit in Week 2 (GEMM tiles of A and B) and Week 4 (FlashAttention tiles of Q, K, V). One fact the page does not say: one block can use at most 227 KB, and anything above 48 KB per block must be requested explicitly at launch (NVIDIA CUDA programming guide).

Source: artifacts/sources/modal-glossary/occupancy.txt:14-21

## Number 4: L2 cache, 50 MB

```
H100 SXM L2 cache: 50 MB, shared by all 132 SMs.
Data reused across blocks stays on chip if the reused set fits in 50 MB.
```

The glossary pages saved here do not state the L2 size; 50 MB is from NVIDIA's H100 architecture whitepaper.

**L2 limits how much data can be reused across blocks without going back to HBM.** It matters for GEMM tile ordering (Week 2 Tuesday, "grouped tile ordering for L2 reuse").

It also matters for **honest benchmarks**. `reduce.py` times 20 launches on the same array. At 268 MB the array cannot fit in 50 MB, so every launch really reads HBM. If you shrink `N` to 4M floats (16 MB, the size in Harris's slides), the array fits in L2, launches 2-20 read from cache, and you will see more than 3.35 TB/s. That number is fake. Keep any bandwidth test well above 50 MB.

Source: tutor

## Number 5: SM count, 132

```
An H100 SXM GPU draws at most 700 W and has 132 SMs, each of which has four Warp Schedulers that can
each issue instructions to 32 threads (aka a warp ) in parallel per clock cycle, for a total of 128
× 132 > 16,000 parallel threads running at about 5 cW apiece.
Note that this is truly parallel: each of the 16,000 threads can make progress with each clock
cycle.
GPU SMs also support a large number of concurrent threads -- threads of execution whose instructions
are interleaved.
A single SM on an H100 can concurrently execute up to 2048 threads split across 64 thread groups of
32 threads each.
With 132 SMs, that's a total of over 250,000 concurrent threads.
```

**SM count limits how many blocks run at the same time (one "wave").** With 128-thread blocks, one SM holds 2048 / 128 = 16 blocks, so one wave is 16 × 132 = 2112 blocks.

Predictions for `reduce.py`:
- Kernel 1 launches N / 128 = 524,288 blocks, about 248 waves. Lots of block start-up overhead.
- Kernel 7 launches `132 * 4` = 528 blocks, 4 per SM, which is 4 × 4 = 16 warps per SM out of 64 possible (25% occupancy). Harris says this is enough on his 2007 GPU. Friday's profile tests whether it is enough on an H100 (see `w1-fri-ncu-rmsnorm-companion.md`, the Little's Law section).

Source: artifacts/sources/modal-glossary/streaming-multiprocessor.txt:18-27

## The ridge point: 295 FLOPs per byte

```
The point on the boundary, i.e. where the diagonal and horizontal roof meet, is called the "ridge
point".
Its x-coordinate is the minimum arithmetic intensity required to be able to escape the memory
bottleneck .
```

Compute it yourself: ridge = compute roof / memory roof = 989e12 FLOP/s ÷ 3.35e12 B/s = **295 FLOPs per byte**. A kernel must do 295 BF16 tensor-core FLOPs for every byte it moves before memory stops being its limit.

The reduction does 1 add per 4-byte float: 0.25 FLOPs per byte, about 1200 times below the ridge. It is memory-bound by a huge margin. So the only question for Tuesday is "what fraction of 3.35 TB/s do I reach", never "how many FLOPs".

Source: artifacts/sources/modal-glossary/roofline-model.txt:20-23

## The ridge depends on which unit does the math

```
The compute and memory roofs need only be derived once per subsystem (though importantly they vary
depending on the subsystem, not just the system; Tensor Cores have more FLOPS than CUDA Cores ).
```

The reduction adds FP32 numbers on CUDA cores, not tensor cores. H100 SXM FP32 on CUDA cores is about 67 TFLOP/s (NVIDIA datasheet), so that ridge is 67 / 3.35 ≈ **20 FLOPs per byte**. The reduction (0.25) is still 80 times below it. Same conclusion, but write both ridges in your notes: in Week 2 your CUDA-core GEMM hits the 20 roof and your tensor-core GEMM hits the 295 roof.

Source: artifacts/sources/modal-glossary/roofline-model.txt:27-28
