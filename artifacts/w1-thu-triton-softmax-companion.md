# Companion: Week 1 Thursday, Triton fused softmax, then make it one pass

The tutorial source: `sources/triton-02-fused-softmax.py` (snapshot of `python/tutorials/02-fused-softmax.py` from the Triton repo, 233 lines). The rendered page is https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html; the line numbers below are for the saved file.

The plan's task: build online softmax with one row per block, starting from this tutorial, and check it against `torch.softmax`. Read Wednesday's companion first; this one assumes the `(m, d)` update.

Source: tutor

## Before you open the file

```
**Rule for every kernel:** before you open the file, write down what it computes, how work splits across blocks and warps, where each tensor lives (global, shared, registers), and whether it is memory-bound or compute-bound. Then read the code and mark where you were wrong.
```

Write your four answers for the tutorial kernel now. Mine are in the next-but-one section, so you can mark yourself.

Source: README.md:7-7

## The motivation: unfused PyTorch reads and writes 4 times too much

```
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
```

Each PyTorch line is a separate kernel. Each kernel reads its input from HBM and writes its output back to HBM, because separate kernels cannot share registers. For an M × N matrix that is 8MN element accesses. A fused kernel reads `x` once and writes `y` once: 2MN. So the expected speed-up is about 4x, and since softmax is memory-bound (Wednesday companion, Part 0), access count is speed.

This is the general lesson of the tutorial: **for memory-bound work, fusing steps into one kernel removes the HBM round trips between them.** RMSNorm in vLLM (Friday) and FlashAttention (Week 4) are both this idea.

Source: artifacts/sources/triton-02-fused-softmax.py:48-58

## The kernel, line by line

```
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

Translation to CUDA terms:

- **A Triton "program" is a CUDA block.** `tl.program_id(0)` is `blockIdx.x`, `tl.num_programs(0)` is `gridDim.x`. You never see threads; Triton spreads each vector op over `num_warps × 32` threads for you.
- **The `for row_idx` loop is a grid-stride loop**, the same shape as Harris kernel 7: a fixed number of programs, each taking rows `pid, pid + grid, pid + 2·grid, ...`.
- **`tl.arange(0, BLOCK_SIZE)` is a vector of column indices.** `BLOCK_SIZE` must be a power of two, so it is rounded up past `n_cols`, and `mask` switches off the extra lanes.
- **`other=-float('inf')` is not a detail.** Masked lanes load `-inf`, so they never win `tl.max`, and `exp(-inf - m) = 0`, so they add nothing to `tl.sum`. If they loaded 0 instead, a row of all-negative numbers would get the wrong max.
- **`tl.max` and `tl.sum` are block reductions.** The compiler emits a Harris-style tree: warp shuffles inside each warp, then shared memory across warps. Tuesday's whole lecture is one function call here.
- **`tl.exp` is approximate** (like CUDA `__expf`). Results differ from PyTorch in the last bits, which matters for the tolerance in your test.

Source: artifacts/sources/triton-02-fused-softmax.py:87-109

## My answers to the four questions

```
Computes:   y = softmax(x) along each row of an M x N matrix.
Work split: one program (block) per row at a time, persistent grid-stride over rows;
            the row's columns are spread over num_warps x 32 threads.
Lives in:   x, y in global memory (HBM). The whole row in registers during the math;
            shared memory only as scratch inside tl.max / tl.sum.
Bound:      memory-bound; one HBM read of x, one HBM write of y, a few FLOPs per element.
```

Note the second line: the whole row must fit in registers at once. That is the limit the tutorial title admits: "those whose rows can fit in the GPU's SRAM". Very long rows (for example a 256k-token vocabulary) spill or fail to compile.

Source: tutor

## The launch: how many programs to start

```
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
```

The host code compiles the kernel once (`warmup`), reads how many registers per thread it uses (`n_regs`), and computes how many programs fit on one SM at once: register file size ÷ registers per program, capped by shared memory. Then it launches exactly enough programs to fill every SM once (`NUM_SM * occupancy`), and each one loops over rows. These are **persistent programs**: started once, never replaced.

This is the same arithmetic as the Modal occupancy page example from Monday (`w1-mon-h100-numbers.md`, number 3). On an H100, `NUM_REGS` is 65,536 per SM; with 8 warps and, say, 32 registers per thread, one program uses 32 × 32 × 8 = 8192 registers, so 8 programs per SM and 1056 programs total.

Source: artifacts/sources/triton-02-fused-softmax.py:166-174

## Thursday's change: what "one pass" means here

```
Tutorial (row fits in registers):
  load whole row once -> max over registers -> exp, sum over registers -> store
  HBM traffic: 1 read + 1 write. Already minimal. Row length limited by registers.

Online, one row per block, chunks of BLOCK columns (any row length):
  loop 1 over chunks: load chunk, update running (m, d) with Wednesday's rescale
  loop 2 over chunks: load chunk again, store exp(x - m) / d
  HBM traffic: 2 reads + 1 write.

Safe softmax in chunks (what you avoid): max loop, sum loop, write loop
  HBM traffic: 3 reads + 1 write.
```

So the online version is not faster than the tutorial when the row fits. It is what lets you handle rows that do not fit, with one read fewer than the chunked safe version. That matches the caveat at the end of the Wednesday companion. To see the difference, test rows longer than one chunk: for example `BLOCK = 1024` and `n_cols = 65536`.

Plan of the kernel, in steps (write the Triton yourself):
1. `pid = tl.program_id(0)` picks the row. Grid = `(n_rows,)`. No persistent loop needed for "one row per block".
2. Running state: a scalar max `m` starting at minus infinity and a scalar sum `d` starting at `0.0`.
3. Loop 1: for each chunk, load with `mask` and `other=-inf`, take the chunk max, form the new max, rescale `d` by `exp(m_old - m_new)`, add the chunk's `sum(exp(x - m_new))`.
4. Loop 2: reload each chunk, store `exp(x - m) / d` with the same mask.

Source: tutor

## Bugs you are likely to hit

```
1. d = 0 instead of d = 0.0
   Triton loop-carried variables must keep one type; an int start that becomes
   float inside the loop is a compile error (or a silent int).
2. tl.max vs tl.maximum
   tl.max(x, axis=0) reduces a vector to a scalar; tl.maximum(a, b) is elementwise.
   The new running max is tl.maximum(m, tl.max(chunk, axis=0)).
3. exp(-inf - (-inf)) = exp(NaN)
   If a whole chunk is masked and m is still -inf, the rescale is NaN.
   Cannot happen when every chunk has at least one real column and real values are finite,
   but it is the classic bug when a mask sets all scores to -inf (Week 4 attention).
4. Forgetting the mask on the store in loop 2: you write past the end of the row.
```

Item 1 is from my experience with Triton's type rules, not from the tutorial; if the compiler accepts `d = 0`, check that `d` is a float afterwards.

Source: tutor

## Checking correctness

```
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

The tutorial tests a shape with an odd row count and a column count that is not a power of two (781), so the mask is exercised. For your online kernel, add three more tests:

- **Many chunks:** `n_cols = 65536` with `BLOCK = 1024`, so loop 1 runs 64 times and the rescale actually fires.
- **Large values:** `x * 100`. Logits near 300 overflow naive softmax (Wednesday, `exp(89)` is already infinite). Your kernel must still match.
- **Max in the last chunk:** set `x[:, -1] = 50`. The running max then changes on the last chunk, so every earlier partial sum must be rescaled.

If `allclose` fails by tiny amounts, print the largest absolute error before you change the tolerance; `tl.exp` is approximate, but errors above about 1e-5 for FP32 inputs mean a real bug.

Source: artifacts/sources/triton-02-fused-softmax.py:186-190

## Measuring bandwidth

```
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
```

The tutorial counts 2 accesses per element (1 read, 1 write): useful bytes, not actual bytes. Use the same formula for your kernel so the numbers compare, but remember the online version really moves 3 accesses per element when the row spans several chunks.

A prediction to write down before you run: a 65536-column FP32 row is 256 KB. With about one program per row in flight per SM slot, a few hundred rows in flight is tens of MB, which fits in the 50 MB L2. So loop 2's re-read may mostly hit L2, and the measured "useful" bandwidth may come close to the tutorial's. This is a guess; Nsight Compute on Friday can confirm it with the L2 hit rate.

Source: artifacts/sources/triton-02-fused-softmax.py:223-223
