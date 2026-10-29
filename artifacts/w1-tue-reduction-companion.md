# Reading guide: "Optimizing Parallel Reduction in CUDA" (Mark Harris)

The slides: `sources/harris-reduction.pdf` (38 slides, one per PDF page; saved from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf). Each section below is one slide, quotes a line from that slide, and ends with the page number, so you can keep the PDF and this guide side by side.

This is a copy of `notes/w1-reduction-companion.md` with the slide quotes and page links added. The explanations are the same.

The code:
- **`kernels/w1/reduce.py`**: all 7 kernels in one file, runnable on a Modal H100 with `modal run kernels/w1/reduce.py`. It checks each answer and prints time and bandwidth. Kernel numbers match the slides. Code slides below say which lines of `reduce.py` hold your H100 version.
- **NVIDIA's official version**: [`reduction_kernel.cu`](https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu) in NVIDIA's cuda-samples. Its names are off by one: `reduce0` is the slides' Kernel 1, `reduce6` is Kernel 7. It also uses cooperative groups (`cg::sync(cta)` is the same as `__syncthreads()`), and adds extra versions (`reduce7`, `cg_reduce`) that are not in the slides. The slides themselves have the same off-by-one: slide 7 names Kernel 1 `reduce0`.

The slides were made for a live talk. The speaker said most of the explanation out loud, and the slides only show the code and the results. This guide puts back what the speaker said. Read **Part 0** first, then read one section per slide.

```
Kernel N here = "Kernel N" / "Reduction #N" in the slides (slide numbers in comments).
Changes from the slides, so the code is correct on an H100:
  - float instead of int.
  - Kernels 5-7 do the last warp with __shfl_down_sync, not the volatile
    warp-synchronous trick (unsafe since Volta, see companion slide 22).
```

Source: kernels/w1/reduce.py:5-9

## Part 0: What you need to know before slide 1

### The task

You have 4 million integers in GPU memory. You want their sum.

On a CPU, this is one loop:

    int sum = 0;
    for (int i = 0; i < n; i++) sum += a[i];

The loop does one addition at a time, 4 million times in a row.

A GPU has thousands of threads that can run at the same time. You want all of them to add numbers at once. But in the end, there must be **one** number. So the threads must combine their partial results. The whole talk is about how to do that combining fast.

This operation (combine many values into one value with `+`, `max`, or similar) is called a **reduction**.

### How a GPU program is organized

A **kernel** is a function that runs on the GPU. You start it from the CPU like this:

    myKernel<<<numBlocks, threadsPerBlock, sharedBytes>>>(arguments);

This starts `numBlocks × threadsPerBlock` threads. **Every thread runs the same function.** The only difference between threads is their ID numbers:

- `threadIdx.x`: the thread's number inside its block (0, 1, 2, ... up to `threadsPerBlock - 1`).
- `blockIdx.x`: the block's number (0, 1, 2, ... up to `numBlocks - 1`).
- `blockDim.x`: the number of threads per block (the same for every thread).

The threads use these numbers to decide which part of the data to work on.

**Example.** 16 numbers, 2 blocks of 8 threads. Each thread computes `i = blockIdx.x * blockDim.x + threadIdx.x`:

| | Thread 0 | Thread 1 | ... | Thread 7 |
|---|---|---|---|---|
| Block 0 | i = 0 | i = 1 | ... | i = 7 |
| Block 1 | i = 8 | i = 9 | ... | i = 15 |

So block 0 works on elements 0-7 and block 1 works on elements 8-15.

### Blocks: what threads in a block can and cannot do

Threads are grouped into **blocks**. One block runs on one **SM** (streaming multiprocessor, one of the GPU's processors; the G80 GPU in the slides has 16, an H100 has 132).

Threads **in the same block** can:
1. Share a small, fast memory called **shared memory**.
2. Wait for each other with `__syncthreads()`. When a thread reaches this line, it stops until every thread in its block reaches the same line.

Threads **in different blocks** can do neither. They cannot see each other's shared memory, and they cannot wait for each other.

### Two kinds of memory

| | Global memory | Shared memory |
|---|---|---|
| Where | The GPU's main memory chips (HBM on an H100) | Inside each SM |
| Size | Gigabytes | Tens of kilobytes per SM |
| Speed | Slow to reach (hundreds of clock cycles) | Fast (a few cycles) |
| Who sees it | All threads in all blocks | Only the threads of one block |
| In code | Normal pointers passed to the kernel, like `g_idata` | Arrays declared `__shared__`, like `sdata` |

The input array is in global memory. The usual plan: each block copies its part of the input into shared memory, does the work there, and writes one result back to global memory.

### Warps: how the hardware actually runs threads

The hardware does not run threads one by one. It runs them in groups of **32 consecutive threads**, called a **warp**. In a block of 128 threads:

- Warp 0 = threads 0-31
- Warp 1 = threads 32-63
- Warp 2 = threads 64-95
- Warp 3 = threads 96-127

The hardware issues **one instruction for the whole warp**. All 32 threads do that instruction at the same time.

This matters for `if` statements. Suppose that in one warp, the `if` is true for 1 thread and false for 31. The warp still spends the full time running the body of the `if`. The 31 threads with a false condition are switched off during that time; they do nothing useful, but they do not save any time either.

**So the cost of a step is the number of warps that have at least one thread with work, not the number of threads with work.** This is the most important idea for slides 9-12.

Two threads in the same warp that take different sides of an `if` is called **divergence**.

Source: tutor

## Slide 1: Title

**From the slide:**

```
Optimizing Parallel Reduction in CUDA
```

Mark Harris worked on NVIDIA's developer technology team. This talk is from 2007 and is the standard first example of how to make a GPU kernel fast.

Source: artifacts/sources/harris-reduction.pdf p. 1

## Slide 2: Parallel Reduction

**From the slide:**

```
We’ll walk step by step through 7 different versions
```

The speaker gives three reasons for the topic:

- **"Common and important."** Many programs need sums, maximums and minimums of large arrays. Inside the kernels later in this plan, reductions are everywhere: LayerNorm sums the squares of a row; softmax finds the maximum of a row and then sums it.
- **"Easy to implement, harder to get right."** A first version takes 15 lines. A fast version takes much more thought. The first version in the talk is 30 times slower than the last one.
- **"Seven different versions."** The talk writes the kernel 7 times. Each version fixes one performance problem. The problems are the ones you meet in almost every GPU kernel, so the talk is a tour of them.

Source: artifacts/sources/harris-reduction.pdf p. 2

## Slide 3: Parallel Reduction (the tree)

**From the slide:**

```
But how do we communicate partial results between
thread blocks?
```

**On this slide:** 8 numbers at the top, added in pairs, level by level, until one number (25) is left.

**How the tree works.** Start with 8 numbers: `3 1 7 0 4 1 6 3`.

- Level 1: add neighbours in pairs. `3+1=4`, `7+0=7`, `4+1=5`, `6+3=9`. Now there are 4 numbers. These 4 additions do not depend on each other, so 4 threads can do them **at the same time**.
- Level 2: `4+7=11`, `5+9=14`. 2 numbers, 2 threads.
- Level 3: `11+14=25`. 1 number, 1 thread.

A CPU loop needs 7 additions in a row. The tree also does 7 additions, but in only **3 steps**, because the additions in one level happen at the same time. In general, N numbers need log2(N) steps: 128 numbers need 7 steps, 1024 numbers need 10.

**Why many blocks?** The tree runs inside one block, because the threads must share the numbers (shared memory) and wait for each other between levels (`__syncthreads()`). But:

1. A block can have at most 512 threads on the G80 (1024 on an H100). So one block can sum only about that many numbers.
2. One block runs on one SM. The rest of the GPU would do nothing.

So the array is split into pieces. Each block sums its piece with a tree and produces **one partial sum**. With 4 million numbers and 128 threads per block, that is 32,768 blocks and 32,768 partial sums.

**The open question at the bottom of the slide:** the 32,768 partial sums are in 32,768 different blocks. How do we add them together, when blocks cannot share memory or wait for each other?

Source: artifacts/sources/harris-reduction.pdf p. 3

## Slide 4: Problem: Global Synchronization

**From the slide:**

```
But CUDA has no global synchronization. Why?
```

**The idea the slide starts with.** "Let every block write its partial sum to global memory. Then all blocks wait until every block has finished. Then one block adds the partial sums." The "all blocks wait" part is called a **global synchronization**: a `__syncthreads()` for the whole GPU instead of one block.

**CUDA has no such instruction. Why?** The slide gives two reasons. The second one needs explanation.

The GPU does not run all 32,768 blocks at the same time. It runs as many as fit on its SMs. On the G80, roughly 16 SMs × 8 blocks = 128 blocks run at once. The other blocks wait in a queue. When a running block finishes, its SM takes the next block from the queue.

Now suppose there was a global wait. The first 128 blocks start, compute their partial sums, and reach the wait. They stop there until all 32,768 blocks arrive. But the other 32,640 blocks are still in the queue. They can start only when a running block finishes, and no running block will finish, because they are all waiting. Every block waits forever. This is a **deadlock**.

The only way to avoid it is to start no more blocks than fit on the GPU at once. That is the slide's "no more than # multiprocessors × # resident blocks / multiprocessor". This limits how you can write your program, so CUDA does not offer it.

**The solution on the slide: use two kernels.** When a kernel finishes, every one of its blocks has finished. So:

1. Kernel 1: each block computes its partial sum and writes it to global memory.
2. Kernel 1 ends. Now all partial sums are in global memory.
3. Kernel 2: reads the partial sums and adds them.

The end of one kernel and the start of the next is the global wait we wanted. The slide says this costs very little.

**Today:** CUDA now has a global wait inside a kernel (cooperative groups `grid.sync()`), but it has exactly the restriction above: all blocks must fit on the GPU at once. Another common option today: each block does `atomicAdd(&total, partialSum)`, which adds its partial sum directly to one global number.

Source: artifacts/sources/harris-reduction.pdf p. 4

## Slide 5: Kernel Decomposition

**From the slide:**

```
In the case of reductions, code for all levels is the
same
```

**On this slide:** 64 numbers in 8 groups of 8. "Level 0: 8 blocks" sums each group into one number (each group is `3 1 7 0 4 1 6 3`, so each sum is 25). "Level 1: 1 block" sums the 8 results.

**The point:** level 1 is the same problem as level 0, only smaller. So it uses **the same kernel**, just started again with the output of the previous level as its input.

**With real numbers** (4,194,304 numbers, 128 threads per block, each block sums 128 numbers):

| Launch | Input numbers | Blocks | Output numbers |
|---|---|---|---|
| 1 | 4,194,304 | 32,768 | 32,768 |
| 2 | 32,768 | 256 | 256 |
| 3 | 256 | 2 | 2 |
| 4 | 2 | 1 | 1 |

Almost all the time is spent in launch 1, because it reads all 16 MB of input. The later launches handle very little data. The timings in the rest of the talk are for this whole process.

Source: artifacts/sources/harris-reduction.pdf p. 5

## Slide 6: What Is Our Optimization Goal?

**From the slide:**

```
1 flop per element loaded (bandwidth-optimal)

```

**The question:** how do we know when the kernel is fast enough? Compare it with the limit of the hardware.

**There are two limits.** A GPU can only do math so fast (measured in GFLOP/s, billions of operations per second), and it can only read memory so fast (measured in GB/s). Which limit matters depends on the kernel:

- If a kernel does a lot of math for each number it reads, the math is the limit. Measure GFLOP/s.
- If a kernel does little math for each number, reading the numbers is the limit. Measure GB/s.

**Reduction does almost no math.** Each number is read once and used in one addition. The GPU can add much faster than it can read numbers from memory. So the limit is memory. The goal: read the input as close to the memory's top speed as possible.

**How to measure it:** bandwidth = bytes read ÷ time. For example, 2^22 integers × 4 bytes = 16.8 MB. If the kernel takes 8 ms, bandwidth = 16.8 MB ÷ 0.008 s = 2.1 GB/s.

**The G80's top memory speed:**
- The memory bus is 384 bits wide = 48 bytes per transfer.
- The memory clock is 900 MHz, and DDR memory makes 2 transfers per clock = 1,800 million transfers per second.
- 48 bytes × 1,800 million per second = **86.4 GB/s**.

Every result in the talk should be compared with 86.4 GB/s. For your H100 work, the number is 3,350 GB/s.

Source: artifacts/sources/harris-reduction.pdf p. 6

## Slide 7: Reduction #1: Interleaved Addressing (the code)

**From the slide:**

```
for(unsigned int s=1; s < blockDim.x; s *= 2) {
if (tid % (2*s) == 0) {
sdata[tid] += sdata[tid + s];
```

**On this slide:** the first version of the kernel.

Walk through it with a small block of **8 threads** (the talk uses 128, but 8 is easier to follow). Suppose this is block 1, so it handles elements 8-15.

    __global__ void reduce0(int *g_idata, int *g_odata) {

`g_idata` is the input array in global memory. `g_odata` is the output array: one partial sum per block.

      extern __shared__ int sdata[];

An array in shared memory. Its size is not written here; it comes from the third number in `<<< >>>` when the kernel starts (128 threads × 4 bytes = 512 bytes). Each block gets its own copy.

      unsigned int tid = threadIdx.x;
      unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
      sdata[tid] = g_idata[i];

Each thread copies one number from global memory into shared memory. Thread 3 of block 1: `tid = 3`, `i = 1×8 + 3 = 11`, so it copies `g_idata[11]` into `sdata[3]`.

Why copy at all? The tree reads and writes each position several times. Shared memory is much faster than global memory for that.

      __syncthreads();

Wait until every thread has finished its copy. Without this line, thread 0 might try to add `sdata[1]` before thread 1 has written it.

      for (unsigned int s=1; s < blockDim.x; s *= 2) {
         if (tid % (2*s) == 0) {
             sdata[tid] += sdata[tid + s];
         }
         __syncthreads();
      }

This is the tree. `s` is the **stride**: the distance between the two numbers being added. It is 1, then 2, then 4.

`tid % (2*s)` is the remainder of `tid` divided by `2*s`. So `tid % (2*s) == 0` means "tid is a multiple of 2s". Only those threads add.

With 8 threads and input values `a0 ... a7` in `sdata`:

| Step | s | Threads that add (multiples of 2s) | What they do |
|---|---|---|---|
| 1 | 1 | 0, 2, 4, 6 | `sdata[0] += sdata[1]`, `sdata[2] += sdata[3]`, `sdata[4] += sdata[5]`, `sdata[6] += sdata[7]` |
| 2 | 2 | 0, 4 | `sdata[0] += sdata[2]`, `sdata[4] += sdata[6]` |
| 3 | 4 | 0 | `sdata[0] += sdata[4]` |

After step 3, `sdata[0]` holds the sum of all 8 values.

The `__syncthreads()` inside the loop is needed because each step reads values written in the step before. In step 2, thread 0 reads `sdata[2]`, which thread 2 wrote in step 1.

      if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }

Thread 0 writes the block's sum to the output array, at the position of this block. The next kernel launch (slide 5) sums these outputs.

**Your H100 version:** `kernels/w1/reduce.py` lines 25-37 (`reduce1`).

Source: artifacts/sources/harris-reduction.pdf p. 7

## Slide 8: Interleaved Addressing (the diagram)

**From the slide:**

```
Parallel Reduction: Interleaved Addressing
```

**On this slide:** the same algorithm drawn for 16 values. Each row shows the contents of shared memory after a step. The "Thread IDs" row shows which threads do an addition in that step.

Check the first step yourself. The starting values are:

    position:  0  1  2   3  4   5  6  7   8   9  10 11  12  13  14 15
    value:    10  1  8  -1  0  -2  3  5  -2  -3   2  7   0  11   0  2

Step 1 (stride 1): threads 0, 2, 4, ..., 14 each add the value one position to the right.
- Thread 0: position 0 = 10 + 1 = **11**
- Thread 2: position 2 = 8 + (-1) = **7**
- Thread 4: position 4 = 0 + (-2) = **-2**
- ...and so on. This matches the slide's second row: `11 1 7 -1 -2 -2 8 5 ...`

The odd positions keep their old values. Nobody reads them again, so it does not matter.

Follow position 0 to the end: 10 → 11 (step 1) → 18 (step 2, adds position 2) → 24 (step 3, adds position 4) → **41** (step 4, adds position 8). 41 is the sum of all 16 values.

**Look at which threads work:** 0, 2, 4, 6, ...; then 0, 4, 8, 12; then 0, 8; then 0. The working threads are **spread out**, with gaps between them. Slide 9 explains why that is slow.

Source: artifacts/sources/harris-reduction.pdf p. 8

## Slide 9: Reduction #1: the problem

**From the slide:**

```
Problem: highly divergent
```

**On this slide:** the same code, with a note: "highly divergent warps are very inefficient, and % operator is very slow."

**Problem 1: divergence.** Go back to Part 0: a warp is 32 threads, and a step costs one unit of time **for each warp that has at least one thread with work**.

With 128 threads there are 4 warps. Count, for each step, which warps contain a working thread:

| s | Threads that add | How many add | Warps that must run |
|---|---|---|---|
| 1 | 0, 2, 4, ..., 126 | 64 | all 4 (every warp has even-numbered threads) |
| 2 | 0, 4, 8, ..., 124 | 32 | all 4 |
| 4 | 0, 8, 16, ..., 120 | 16 | all 4 |
| 8 | 0, 16, 32, ..., 112 | 8 | all 4 |
| 16 | 0, 32, 64, 96 | 4 | all 4 (one thread in each warp) |
| 32 | 0, 64 | 2 | 2 |
| 64 | 0 | 1 | 1 |
| | | **127 additions** | **23 warp-steps** |

At s = 16, only 4 threads do work, but all 4 warps must run, each with 31 of 32 threads switched off. The working threads are spread across all warps, so no warp can stop early.

**Problem 2: the `%` operator.** A GPU has no fast hardware for integer division, and `%` (remainder) needs a division. It compiles into many instructions. Every thread computes it in every step.

Source: artifacts/sources/harris-reduction.pdf p. 9

## Slide 10: Performance for 4M element reduction

**From the slide:**

```
8.054 ms

```

**On this slide:** a table with one row. Kernel 1 takes 8.054 ms for 2^22 integers, which is 2.083 GB/s. Note: block size is 128 threads for all tests.

(The PDF shows "222 ints". It means 2 to the power 22 = 4,194,304. The superscript was lost.)

**What the number means:** 2.083 GB/s ÷ 86.4 GB/s = **2.4%** of the memory's top speed. The memory could deliver the data about 40 times faster than the kernel uses it. So memory is not what slows this kernel down. The kernel is busy with its own instructions: the `%`, the `if`, the warps running with most threads switched off.

Each row added in the later tables is one more version. Watch the percentage go up.

Source: artifacts/sources/harris-reduction.pdf p. 10

## Slide 11: Reduction #2: Interleaved Addressing (the code change)

**From the slide:**

```
int index = 2 * s * tid;

if (index < blockDim.x) {
```

**On this slide:** the old loop at the top, the new loop at the bottom.

New loop:

    for (unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

**The additions are exactly the same as before. Only the thread that does each addition changes.**

In version 1, the thread number was the position: thread 2 worked on position 2. In version 2, each thread computes a position `index = 2 × s × tid`:

| s | Positions to add (same as before) | Version 1: which threads | Version 2: which threads |
|---|---|---|---|
| 1 | 0, 2, 4, 6, ... | 0, 2, 4, 6, ... | 0, 1, 2, 3, ... |
| 2 | 0, 4, 8, 12, ... | 0, 4, 8, 12, ... | 0, 1, 2, 3, ... |
| 4 | 0, 8, 16, ... | 0, 8, 16, ... | 0, 1, 2, ... |

Check: at s = 2, thread 1 computes `index = 2 × 2 × 1 = 4`. So thread 1 adds position 6 into position 4. In version 1, thread 4 did that.

`if (index < blockDim.x)` switches off the threads whose position would be past the end of the array.

The `%` is gone too.

**Your H100 version:** `kernels/w1/reduce.py` lines 39-52 (`reduce2`).

Source: artifacts/sources/harris-reduction.pdf p. 11

## Slide 12: Interleaved Addressing (diagram), and a new problem

**From the slide:**

```
New Problem: Shared Memory Bank Conflicts
```

**On this slide:** the same 16 values, the same results as slide 8 (41 at the end), but the thread IDs are now 0-7, then 0-3, then 0-1, then 0. At the bottom: "New Problem: Shared Memory Bank Conflicts".

**Why divergence is fixed.** The working threads are now always the lowest-numbered ones, packed together. Count warps again with 128 threads:

| s | Threads that add | Warps that must run |
|---|---|---|
| 1 | 0-63 | 2 (warps 0 and 1 are full; warps 2 and 3 have no work and skip) |
| 2 | 0-31 | 1 |
| 4 | 0-15 | 1 |
| 8 | 0-7 | 1 |
| 16 | 0-3 | 1 |
| 32 | 0-1 | 1 |
| 64 | 0 | 1 |
| | | **8 warp-steps** (version 1: 23) |

Same 127 additions, about a third of the warp work.

**The new problem: bank conflicts.** To understand this, you need to know how shared memory is built.

Shared memory is split into **32 banks**. Consecutive 4-byte numbers go to consecutive banks, and then it wraps around:

    sdata index:  0   1   2  ...  31   32  33  ...  63   64 ...
    bank:         0   1   2  ...  31    0   1  ...  31    0 ...

(Bank = index mod 32.)

Each bank can deliver **one number per clock cycle**. When the 32 threads of a warp read shared memory:
- If all 32 threads read from 32 **different** banks: 1 cycle.
- If 2 threads read **different** numbers in the **same** bank: that bank must deliver them one after the other: 2 cycles. This is a **2-way bank conflict**.
- If 4 threads hit the same bank: 4 cycles. And so on.

Now look at version 2. At s = 1, thread `t` reads `sdata[2t]`:

| Thread | 0 | 1 | 2 | ... | 15 | 16 | 17 | ... | 31 |
|---|---|---|---|---|---|---|---|---|---|
| Index read | 0 | 2 | 4 | ... | 30 | 32 | 34 | ... | 62 |
| Bank | 0 | 2 | 4 | ... | 30 | **0** | **2** | ... | 30 |

Threads 0 and 16 both use bank 0. Threads 1 and 17 both use bank 2. Every bank that is used is used twice, and the odd banks are not used at all. 2-way conflict: the read takes twice as long.

At s = 2, thread `t` reads `sdata[4t]`: banks 0, 4, 8, ..., each used by 4 threads. 4-way conflict. The larger the stride, the worse it gets.

So version 2 fixed one problem (divergence) and the way it spaces out its reads created another one.

(Small detail: the G80 had 16 banks and checked conflicts per half-warp of 16 threads. The idea is the same. The numbers above are for current GPUs.)

Source: artifacts/sources/harris-reduction.pdf p. 12

## Slide 13: Performance, Kernel 2

**From the slide:**

```
4.854 GB/s
```

**New row:** 3.456 ms, 4.854 GB/s. "Step speedup" 2.33x means 2.33 times faster than the version before. "Cumulative speedup" means faster than Kernel 1.

4.854 ÷ 86.4 = **5.6%** of top speed. Better, still far from the limit.

Source: artifacts/sources/harris-reduction.pdf p. 13

## Slide 14: Sequential Addressing (diagram)

**From the slide:**

```
Sequential addressing is conflict free
```

**On this slide:** a new pattern. Instead of adding neighbours first, add the far half first. With 16 values, step 1 has stride 8: thread `t` adds position `t + 8` into position `t`.

Check step 1 with the starting values:

| Thread | Adds | Result in position t |
|---|---|---|
| 0 | 10 + (-2) | 8 |
| 1 | 1 + (-3) | -2 |
| 2 | 8 + 2 | 10 |
| 3 | -1 + 7 | 6 |
| 4 | 0 + 0 | 0 |
| 5 | -2 + 11 | 9 |
| 6 | 3 + 0 | 3 |
| 7 | 5 + 2 | 7 |

This matches the slide's second row: `8 -2 10 6 0 9 3 7`. Positions 8-15 are not used again.

Step 2 (stride 4): threads 0-3 add positions 4-7 into positions 0-3. Step 3 (stride 2): threads 0-1. Step 4 (stride 1): thread 0. The result is again 41 in position 0.

**Why there are no bank conflicts.** At every step, thread `t` reads `sdata[t]` and `sdata[t + s]`. The 32 threads of a warp read 32 **consecutive** positions, which are in 32 **different** banks. Every read takes 1 cycle.

**Divergence is still fixed.** The working threads are still 0, 1, 2, ... packed together.

Source: artifacts/sources/harris-reduction.pdf p. 14

## Slide 15: Reduction #3: Sequential Addressing (the code)

**From the slide:**

```
for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
```

New loop:

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

- The stride starts at half the block size (64 for 128 threads) and gets smaller.
- `s >>= 1` shifts `s` right by one bit, which divides it by 2. So `s` is 64, 32, 16, 8, 4, 2, 1.
- `if (tid < s)`: the first `s` threads work.

This is shorter and simpler than both earlier versions. That often happens: the version with the best memory access pattern is also the clearest.

**Your H100 version:** `kernels/w1/reduce.py` lines 54-66 (`reduce3`).

Source: artifacts/sources/harris-reduction.pdf p. 15

## Slide 16: Performance, Kernel 3

**From the slide:**

```
9.741 GB/s
```

**New row:** 1.722 ms, 9.741 GB/s, 2 times faster than Kernel 2.

9.741 ÷ 86.4 = **11%**. Two problems are fixed. The kernel still uses only a ninth of the memory speed.

Source: artifacts/sources/harris-reduction.pdf p. 16

## Slide 17: Idle Threads

**From the slide:**

```
Half of the threads are idle on first loop iteration!
```

**On this slide:** the version 3 loop and the remark "Half of the threads are idle on first loop iteration!"

**What this means.** The block has 128 threads. Look at what they do:

- Every thread loads one number. All 128 are busy.
- Step 1 (s = 64): threads 0-63 add. **Threads 64-127 do nothing from now on.**
- Step 2 (s = 32): threads 0-31 add. Threads 32-63 do nothing from now on.
- ...and so on.

Half the threads exist only to do one load. A thread is not free: it needs registers, a place on the SM, and scheduling. We launch twice as many threads as the tree can use.

Source: artifacts/sources/harris-reduction.pdf p. 17

## Slide 18: Reduction #4: First Add During Load

**From the slide:**

```
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
```

**The fix:** let each thread load **two** numbers and add them right away. Then each block of 128 threads handles 256 numbers, and you need half as many blocks.

Old start of the kernel:

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];

New start:

    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];

**Example.** Block 0, 128 threads, handles elements 0-255. Thread 5 computes `i = 0 × 256 + 5 = 5`, loads elements 5 and 133, adds them, and stores the sum in `sdata[5]`. Block 1 handles elements 256-511.

The first level of the tree is now done during the load, in the thread's own registers. It needs no shared memory and no `__syncthreads()`.

**Why `i + blockDim.x` and not `i + 1`?** This is about how global memory is read.

Global memory is read in **chunks** of consecutive bytes (for example 128 bytes = 32 integers). When the 32 threads of a warp read 32 consecutive integers, the hardware fetches **one chunk** and gives each thread its number. This is called **coalescing**, and it is the fastest way to read global memory.

- With `i` and `i + 128`: in the first read, threads 0-31 read elements 0-31 (one chunk). In the second read, they read elements 128-159 (one chunk). Every byte fetched is used.
- If instead thread `t` read elements `2t` and `2t + 1`: in the first read, threads 0-31 read elements 0, 2, 4, ..., 62. These span two chunks, and half of each chunk (the odd elements) is fetched and not used in that read.

So the slide keeps each read consecutive across the warp.

**Your H100 version:** `kernels/w1/reduce.py` lines 68-80 (`reduce4`).

Source: artifacts/sources/harris-reduction.pdf p. 18

## Slide 19: Performance, Kernel 4

**From the slide:**

```
0.965 ms 17.377 GB/s
```

**New row:** 0.965 ms, 17.377 GB/s, 1.78 times faster.

17.377 ÷ 86.4 = **20%**.

Source: artifacts/sources/harris-reduction.pdf p. 19

## Slide 20: Instruction Bottleneck

**From the slide:**

```
Therefore a likely bottleneck is instruction overhead
```

**The speaker's reasoning, in full:**

1. At 17 GB/s, the memory is still idle 80% of the time. So memory is not the limit.
2. The math is one addition per number. That is not the limit either.
3. So what takes the time? The **other** instructions: the ones that are not loads, stores, or the useful additions.

Look at what one step of the loop costs each thread:

    for (s = blockDim.x/2; s > 0; s >>= 1) {   // shift, compare, branch
        if (tid < s)                           // compare, branch
            sdata[tid] += sdata[tid + s];      // add for the address, 2 loads, 1 add, 1 store
        __syncthreads();                       // wait for the whole block
    }

Only the `+=` is useful work. Everything else is overhead: computing `s`, testing the loop condition, jumping back, testing `tid < s`, computing the address `tid + s`, and waiting at `__syncthreads()`.

**The fix: unroll the loop.** "Unrolling" means writing out each step of the loop as separate lines, so the loop counter, the loop test and the jump back disappear. Slides 21-28 do this.

Source: artifacts/sources/harris-reduction.pdf p. 20

## Slide 21: Unrolling the Last Warp

**From the slide:**

```
Instructions are SIMD synchronous within a warp
```

**The observation:** when `s` gets to 32 or less, only threads 0-31 are still working. That is exactly one warp (warp 0).

**On the G80, this allowed two shortcuts.** In 2007, the 32 threads of a warp always ran in exact lockstep: the same instruction at the same moment. So:

1. **No `__syncthreads()` needed.** The threads of one warp are always at the same line, so there is nothing to wait for.
2. **No `if (tid < s)` needed.** The warp runs the instruction anyway (Part 0: the cost is per warp). Letting threads with `tid >= s` also do the addition costs nothing. They write wrong values into positions `s` and above, but those positions are never read again.

So the last 6 steps (s = 32, 16, 8, 4, 2, 1) can be written out with no barrier and no test.

**Important: this is no longer true.** Since the Volta generation of GPUs (2017, which includes the H100), the threads of a warp can run out of step with each other. Code that relies on lockstep can read a value before another thread has written it, and give wrong answers. Slide 22 gives the safe modern version.

Source: artifacts/sources/harris-reduction.pdf p. 21

## Slide 22: Reduction #5: Unroll the Last Warp (the code)

**From the slide:**

```
__device__ void warpReduce(volatile int* sdata, int tid) {
```

__device__ void warpReduce(volatile int* sdata, int tid) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }

These are the 6 last steps of the tree, written out, with no barrier and no `if`.

**Why `volatile`?** Without it, the compiler sees six lines that all update `sdata[tid]` and may keep that value in a register, writing it to shared memory only at the end. Then the other threads never see the updated values, and the result is wrong. `volatile` forces every read and every write on every line to go to shared memory.

The main loop now stops at `s > 32` and calls `warpReduce` for the rest:

    for (unsigned int s=blockDim.x/2; s>32; s>>=1) { ... }
    if (tid < 32) warpReduce(sdata, tid);

**The note at the bottom of the slide:** "This saves useless work in all warps, not just the last one." In version 4, all 4 warps ran the loop test and the `if` for every step, even when only warp 0 had work. Now warps 1-3 leave the loop after s = 64 and are done.

**Do not write this version for an H100.** Use the safe modern version. Each thread keeps its value in a register `v` and reads other threads' registers directly:

    // v holds this thread's partial sum
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    // thread 0 of the warp now holds the sum of all 32 values

`__shfl_down_sync(mask, v, offset)` gives each thread the value `v` of the thread `offset` positions higher in the same warp. With offset 16, thread 0 gets thread 16's value, thread 1 gets thread 17's, and so on. After offsets 16, 8, 4, 2, 1, thread 0 has added all 32 values. The `_sync` part includes the synchronization, so it is safe on current GPUs. `0xffffffff` means all 32 threads take part. It also uses no shared memory at all.

More detail: NVIDIA's [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).

**Your H100 version:** `kernels/w1/reduce.py` lines 82-104 (`warpReduce and reduce5`).

Source: artifacts/sources/harris-reduction.pdf p. 22

## Slide 23: Performance, Kernel 5

**From the slide:**

```
0.536 ms 31.289 GB/s
```

**New row:** 0.536 ms, 31.289 GB/s, 1.8 times faster.

31.289 ÷ 86.4 = **36%**. Removing the loop overhead from the last 6 steps almost doubled the speed. This confirms the reasoning of slide 20: the instructions were the limit.

Source: artifacts/sources/harris-reduction.pdf p. 23

## Slide 24: Complete Unrolling

**From the slide:**

```
Luckily, the block size is limited by the GPU to 512 threads
```

**The next step:** the steps before the last warp (s = 64 for a 128-thread block, more for bigger blocks) are still in a loop. Unroll them too.

**The difficulty:** to write out the steps, the compiler must know how many steps there are, which depends on the block size. But the block size is chosen when the kernel starts, not when it is compiled.

**Why this is solvable:** there are few possible block sizes. The block size is a power of 2 (the talk uses only powers of 2), and the G80 allows at most 512 threads. So there are only 10 possibilities: 1, 2, 4, ..., 512. (Today the limit is 1024.)

**The tool:** C++ templates, which let you compile one copy of the kernel for each block size.

Source: artifacts/sources/harris-reduction.pdf p. 24

## Slide 25: Unrolling with Templates

**From the slide:**

```
template <unsigned int blockSize>
__global__ void reduce5(int *g_idata, int *g_odata)
```

template <unsigned int blockSize>
    __global__ void reduce5(int *g_idata, int *g_odata)

`blockSize` is now a **template parameter**. When you write `reduce5<128>`, the compiler makes a copy of the kernel in which `blockSize` is the constant 128. `reduce5<256>` makes another copy with 256. Inside each copy, the compiler knows the block size exactly.

Source: artifacts/sources/harris-reduction.pdf p. 25

## Slide 26: Reduction #6: Completely Unrolled (the code)

**From the slide:**

```
if (blockSize >= 256) {
if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
```

if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64];  } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

Each step of the tree is written out. The tests on `blockSize` (red on the slide) are tests on a constant, so the compiler decides them while compiling and removes them from the program.

**What the compiler keeps for `blockSize = 128`:**
- `blockSize >= 512` is false: line removed.
- `blockSize >= 256` is false: line removed.
- `blockSize >= 128` is true: kept, without the test.
- In `warpReduce`, all six lines are kept (each has a test like `blockSize >= 64`, all true).

So the compiled kernel is:

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32) { /* the six warpReduce lines */ }

No loop, no loop counter, no tests on `s`. Only the additions, the `tid` tests and one barrier.

(The `warpReduce` part has the same 2017 problem as slide 22. On an H100, use the shuffle version.)

**Your H100 version:** `kernels/w1/reduce.py` lines 106-122 (`reduce6`).

Source: artifacts/sources/harris-reduction.pdf p. 26

## Slide 27: Invoking Template Kernels

**From the slide:**

```
Nope, just a switch statement for 10 possible block sizes:
```

**The question on the slide:** "Don't we still need block size at compile time?"

**The answer:** compile all 10 versions, then pick one at run time with a `switch`:

    switch (threads) {
        case 512: reduce5<512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata); break;
        case 256: reduce5<256><<<...>>>(...); break;
        ...
    }

`threads` is the block size chosen at run time. The `switch` starts the copy that was compiled for that size. You get the speed of a compile-time constant with a choice made at run time.

Source: artifacts/sources/harris-reduction.pdf p. 27

## Slide 28: Performance, Kernel 6

**From the slide:**

```
0.381 ms 43.996 GB/s
```

**New row:** 0.381 ms, 43.996 GB/s, 1.41 times faster.

43.996 ÷ 86.4 = **51%**. Half of the memory's top speed.

Source: artifacts/sources/harris-reduction.pdf p. 28

## Slide 29: Parallel Reduction Complexity

**From the slide:**

```
Work Complexity is O(N) – It is work-efficient
```

This slide and the next two use some theory to explain the last improvement. The terms:

**"Log(N) parallel steps, each step S does N/2^S independent ops."** With N numbers, step 1 does N/2 additions, step 2 does N/4, and so on, for log2(N) steps. With N = 8: 4, then 2, then 1, in 3 steps.

**"Step complexity is O(log N)."** The number of steps grows with log N. Doubling the input adds only one step.

**"Performs N-1 operations. Work complexity is O(N). It is work-efficient."** Total additions: 4 + 2 + 1 = 7 = N - 1 for N = 8. A CPU loop also does N - 1 additions. So the parallel version does no extra additions. That is what "work-efficient" means.

**"With P threads physically in parallel, time complexity is O(N/P + log N)."** If the hardware can do P additions at the same time, the N additions take about N/P time, plus log N for the steps of the tree, which must happen one after another.

Source: artifacts/sources/harris-reduction.pdf p. 29

## Slide 30: What About Cost?

**From the slide:**

```
Brent’s theorem suggests O(N/log N) threads
```

**Cost** here means: number of threads × time. It measures how much of the GPU's capacity the algorithm uses, including threads that are waiting and doing nothing.

**With one thread per number** (versions 1-6):
- N threads, for log N steps.
- Cost = N × log N.
- But only N - 1 additions are useful. The rest of the N log N thread-steps are idle threads (slide 17). So the algorithm is **not cost-efficient**: most of the capacity is wasted.

**Brent's theorem suggests a fix:** use fewer threads, and give each one more work.
- Use N / log N threads.
- First, each thread adds log N numbers by itself, in a normal loop. This takes log N time, and no thread is idle.
- Then the threads do the tree on their N / log N partial sums: about log N more steps.
- Cost = (N / log N threads) × (log N time) = N. Cost-efficient.

**Example** with N = 1,024: log N = 10. Use about 100 threads. Each thread first adds about 10 numbers by itself. Then a tree over the 100 partial sums.

This is called **algorithm cascading**: a sequential part (each thread loops) followed by a parallel part (the tree).

Source: artifacts/sources/harris-reduction.pdf p. 30

## Slide 31: Algorithm Cascading

**From the slide:**

```
On G80, best perf with 64-256 blocks of 128 threads
```

**On this slide:** the practical advice.

- Each thread loads and adds **many** numbers in a loop, then the block does the tree in shared memory.
- Brent's theorem says each thread should add about log N numbers (for a block, that is 1024 or 2048 numbers per block instead of 256).
- In practice, the speaker found that **even more** is better, for three reasons:
  1. **Better latency hiding.** A thread in a loop starts its next memory read right after the last one, so many reads are always in progress. (This is the "bytes in flight" idea from Monday's notes: memory reaches full speed only when enough reads are in progress at once.)
  2. **Fewer levels of kernel launches.** Fewer blocks means fewer partial sums, so the next launch (slide 5) has less to do, and fewer launches are needed.
  3. **The last launches are slow.** A launch with only a few blocks uses only a few SMs and still pays the full cost of starting a kernel.
- On the G80, the best result was 64-256 blocks of 128 threads, with each thread adding 1024-4096 numbers.

**Today:** the H100 equivalent of "64-256 blocks" is a small multiple of its 132 SMs.

Source: artifacts/sources/harris-reduction.pdf p. 31

## Slide 32: Reduction #7: Multiple Adds per Thread (the code)

**From the slide:**

```
while (i < n) {
sdata[tid] += g_idata[i] + g_idata[i+blockSize];
i += gridSize;
```

unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();

`gridDim.x` is the number of blocks. `n` is the length of the array (a new kernel argument).

**Example:** 128 threads per block, 64 blocks. Then `gridSize = 128 × 2 × 64 = 16,384`.

- Block 0, thread 0 starts at `i = 0`. It adds elements 0 and 128. Then `i` becomes 16,384, and it adds elements 16,384 and 16,512. Then 32,768 and 32,896. And so on until the end of the array.
- Block 1, thread 0 starts at `i = 256`, and also jumps by 16,384 each time.

In each pass of the loop, the 64 blocks together cover 16,384 consecutive elements (each block takes a 256-element piece). Then all of them move forward by 16,384 together. This is called a **grid-stride loop**.

With 4 million elements, each thread goes through the loop 4,194,304 ÷ 16,384 = 256 times, so it adds 512 numbers before the tree starts.

**Your H100 version:** `kernels/w1/reduce.py` lines 124-145 (`reduce7`).

Source: artifacts/sources/harris-reduction.pdf p. 32

## Slide 33: Reduction #7, the coalescing note

**From the slide:**

```
to maintain coalescing!
```

**On this slide:** the same code, with a note: "gridSize loop stride to maintain coalescing!"

**Why the jumping pattern?** The obvious alternative is to give each thread one **continuous piece** of the array: thread 0 adds elements 0-511, thread 1 adds elements 512-1023, and so on.

Look at what the 32 threads of a warp read **at the same moment** in each case:

| | Thread 0 | Thread 1 | Thread 2 | ... | Thread 31 |
|---|---|---|---|---|---|
| Grid-stride loop, first read | element 0 | element 1 | element 2 | ... | element 31 |
| Continuous pieces, first read | element 0 | element 512 | element 1024 | ... | element 15,872 |

With the grid-stride loop, the warp reads 32 consecutive elements: **one chunk** from memory (slide 18). With continuous pieces, the 32 reads are far apart: **32 separate chunks**, and most of each chunk is fetched and thrown away. The grid-stride loop keeps every read coalesced.

Source: artifacts/sources/harris-reduction.pdf p. 33

## Slide 34: Performance, Kernel 7

**From the slide:**

```
Kernel 7 on 32M elements: 73 GB/s!
```

**New row:** 0.268 ms, 62.671 GB/s, 1.42 times faster. **30 times faster than Kernel 1.**

62.671 ÷ 86.4 = **73%**.

**"Kernel 7 on 32M elements: 73 GB/s!"** With 8 times more data, it reaches 73 ÷ 86.4 = **84%**. Larger inputs do better because the fixed costs (starting the kernels, the tree at the end of each block, the small later launches) are a smaller share of the total time.

Source: artifacts/sources/harris-reduction.pdf p. 34

## Slide 35: Final Optimized Kernel

**From the slide:**

```
while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
```

**On this slide:** the complete Kernel 7. Read it top to bottom and match each part to the slide that introduced it:

| Part of the code | What it does | Slide |
|---|---|---|
| `template <unsigned int blockSize>` | Block size as a compile-time constant | 25 |
| `i = blockIdx.x*(blockSize*2) + tid` and the `while` loop with `gridSize` | Each thread adds many numbers, two per pass, with coalesced reads | 18, 32, 33 |
| `sdata[tid] = 0` before the loop | Start each thread's sum at 0 | 32 |
| The three `if (blockSize >= ...)` lines | The tree steps before the last warp, unrolled | 26 |
| `if (tid < 32) warpReduce(sdata, tid)` | The last 6 steps, no barrier | 22 |
| `if (tid == 0) g_odata[blockIdx.x] = sdata[0]` | Write the block's sum | 7 |

For an H100: replace `warpReduce` with the `__shfl_down_sync` loop from slide 22.

Source: artifacts/sources/harris-reduction.pdf p. 35

## Slide 36: Performance Comparison (the graph)

**From the slide:**

```
7: Multiple elements per
thread (max 64 blocks)
```

**On this slide:** a graph. The horizontal axis is the number of elements, from 131,072 (2^17) to 33,554,432 (2^25). The vertical axis is the time in milliseconds. Both axes are logarithmic: each grid line is 10 times the one below it. There is one line for each kernel.

**How to read it:**
- Every line goes up to the right: more elements take more time.
- The lines are stacked in order: Kernel 1 at the top (slowest), Kernel 7 at the bottom (fastest), at every input size.
- On the right side, the lines are roughly parallel. On a logarithmic axis, that means each version is a fixed number of times faster than the one above it, for large inputs.
- On the left side (small inputs), the lines flatten out. Below some size, the time no longer drops when the input gets smaller, because the fixed costs (starting kernels) dominate.
- Some lines stop early on the right. Those kernels were only measured up to 4 million (or 8 million) elements. Kernel 7 is the only one measured up to 32 million.

Source: artifacts/sources/harris-reduction.pdf p. 36

## Slide 37: Types of Optimization

**From the slide:**

```
11.84x speedup, combined!
```

The speaker splits the 30 times speedup into two kinds:

**Algorithm changes** (which thread reads which address, and how much work each thread does): Kernel 1→2 (2.33x), 2→3 (2.01x), 3→4 (1.78x), 6→7 (1.42x). Multiplied together: **11.84x**.

**Code changes** (unrolling): Kernel 4→5 (1.8x), 5→6 (1.41x). Multiplied together: **2.54x**.

**The lesson:** most of the speed came from changing the algorithm, not from tuning the code. Fix the algorithm first. Unroll afterwards.

Source: artifacts/sources/harris-reduction.pdf p. 37

## Slide 38: Conclusion

**From the slide:**

```
Optimize your algorithm, then unroll loops
```

Each item on the slide, with where you saw it in the talk:

| Item | Where it appeared |
|---|---|
| Memory coalescing | Slides 18, 33: a warp should read consecutive addresses |
| Divergent branching | Slides 9, 12: working threads should be packed into as few warps as possible |
| Bank conflicts | Slides 12, 14: threads of a warp should use different shared-memory banks |
| Latency hiding | Slide 31: keep many memory reads in progress at once |
| Use peak performance to guide optimization | Slide 6, and every results table: compare with 86.4 GB/s |
| Parallel algorithm complexity | Slides 29-30: step count, work, cost |
| Identify the type of bottleneck | Slide 20: memory, math, or instruction overhead |
| Optimize the algorithm, then unroll | Slide 37 |
| Templates to generate optimal code | Slides 24-27 |

Source: artifacts/sources/harris-reduction.pdf p. 38

## Summary: the 7 kernels

**From slide 34, the Kernel 7 row:**

```
0.268 ms 62.671 GB/s
```

| # | Change | Problem it fixes | Time (ms) | GB/s | % of 86.4 GB/s |
|---|---|---|---|---|---|
| 1 | Threads at multiples of 2s add (`tid % (2s)`) | (starting point) | 8.054 | 2.08 | 2.4% |
| 2 | Packed thread numbers (`index = 2*s*tid`) | Divergence, slow `%` | 3.456 | 4.85 | 5.6% |
| 3 | Stride from large to small (`tid < s`) | Bank conflicts | 1.722 | 9.74 | 11% |
| 4 | Each thread loads two numbers and adds them | Half of the threads idle | 0.965 | 17.38 | 20% |
| 5 | Last 6 steps written out | Loop overhead (unsafe on today's GPUs; use shuffles) | 0.536 | 31.29 | 36% |
| 6 | All steps written out, using templates | The rest of the loop overhead | 0.381 | 44.00 | 51% |
| 7 | Each thread adds many numbers (grid-stride loop) | Too many threads with too little work each | 0.268 | 62.67 | 73% |

Source: artifacts/sources/harris-reduction.pdf p. 34

## What this means for Tuesday

```
| Tue | *build* · Reduction, versions 1-3 only from [Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (naive, no divergence, no bank conflicts). Measure bandwidth for each. Skim the slides for versions 4-7; build them only if you have time left. Read with the [slide-by-slide companion](notes/w1-reduction-companion.md). Code for all 7 versions: [`kernels/w1/reduce.py`](kernels/w1/reduce.py). |
```

You build versions 1-3 on an H100. Before you run each one, write down your guess for its GB/s as a percentage of 3,350 GB/s.

The order of the versions should be the same as on the G80. The size of the gaps will be different, because the H100 is very different hardware. I do not know the H100 numbers; finding them is the point of the exercise. On the G80, version 3 reached only 11% of top speed, so expect versions 1-3 to be far from the limit. Versions 4-7 close most of the gap.

Your Monday numbers for the H100 run (from `w1-mon-h100-numbers.md`): `reduce.py` reads 268 MB, so the best possible time is 80 µs, and the 70% target is 2,345 GB/s, or 114 µs. The script prints a "% peak" column against 3,350 GB/s.

Source: README.md:24-24
