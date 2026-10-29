"""The 7 reduction kernels from Mark Harris's slides, runnable on a Modal H100.

Run:  modal run kernels/w1/reduce.py

Kernel N here = "Kernel N" / "Reduction #N" in the slides (slide numbers in comments).
Changes from the slides, so the code is correct on an H100:
  - float instead of int.
  - Kernels 5-7 do the last warp with __shfl_down_sync, not the volatile
    warp-synchronous trick (unsafe since Volta, see companion slide 22).
Each kernel does one pass: n numbers -> one partial sum per block. The script
adds the partial sums on the host to check the answer, and times the pass.
"""
import modal

app = modal.App("reduce")
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
).pip_install("cupy-cuda12x")

BLOCK = 128          # threads per block, as in the slides
N = 1 << 26          # 67M floats = 268 MB
K7_BLOCKS = 132 * 4  # kernel 7 grid: 4 blocks per H100 SM (slide 31)

SRC = r"""
// Kernel 1, slides 7-9: interleaved addressing, divergent branch.
extern "C" __global__ void reduce1(const float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 2, slide 11: strided index, no divergence, bank conflicts.
extern "C" __global__ void reduce2(const float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int index = 2 * s * tid;
        if (index < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 3, slide 15: sequential addressing, no bank conflicts.
extern "C" __global__ void reduce3(const float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Kernel 4, slide 18: first add during load. Each block covers 2 * blockDim elements.
extern "C" __global__ void reduce4(const float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Last 32 values -> 1 inside one warp, with shuffles (replaces slide 22's warpReduce).
__device__ float warpReduce(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// Kernel 5, slides 21-22: unroll the last warp.
extern "C" __global__ void reduce5(const float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = warpReduce(sdata[tid] + sdata[tid + 32]);
        if (tid == 0) g_odata[blockIdx.x] = v;
    }
}

// Kernel 6, slides 24-27: completely unrolled with a template block size.
template <unsigned int blockSize>
__global__ void reduce6(const float *g_idata, float *g_odata) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockSize];
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    if (tid < 32) {
        float v = warpReduce(sdata[tid] + sdata[tid + 32]);
        if (tid == 0) g_odata[blockIdx.x] = v;
    }
}

// Kernel 7, slides 32-35: many elements per thread with a grid-stride loop.
template <unsigned int blockSize>
__global__ void reduce7(const float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    if (tid < 32) {
        float v = warpReduce(sdata[tid] + sdata[tid + 32]);
        if (tid == 0) g_odata[blockIdx.x] = v;
    }
}
"""

# (kernel name, number of blocks). Kernels 4-7 cover 2 elements per thread per load.
KERNELS = [
    ("reduce1", N // BLOCK),
    ("reduce2", N // BLOCK),
    ("reduce3", N // BLOCK),
    ("reduce4", N // (2 * BLOCK)),
    ("reduce5", N // (2 * BLOCK)),
    (f"reduce6<{BLOCK}>", N // (2 * BLOCK)),
    (f"reduce7<{BLOCK}>", K7_BLOCKS),
]


@app.function(gpu="H100", image=image)
def run():
    import cupy as cp

    x = cp.random.rand(N, dtype=cp.float32)
    want = float(x.sum(dtype=cp.float64))
    out = cp.zeros(N // BLOCK, dtype=cp.float32)
    mod = cp.RawModule(code=SRC, name_expressions=[k for k, _ in KERNELS if "<" in k])

    print(f"{N} floats ({N * 4 / 1e6:.0f} MB), {BLOCK} threads per block, H100 peak 3350 GB/s")
    print(f"{'kernel':<14}{'blocks':>9}{'ms':>9}{'GB/s':>8}{'% peak':>8}")
    for name, blocks in KERNELS:
        k = mod.get_function(name)
        args = (x, out, cp.uint32(N)) if name.startswith("reduce7") else (x, out)
        launch = lambda: k((blocks,), (BLOCK,), args, shared_mem=BLOCK * 4)
        launch()  # warm-up and correctness check
        got = float(out[:blocks].sum(dtype=cp.float64))
        assert abs(got - want) / want < 1e-4, f"{name}: got {got}, want {want}"
        t0, t1 = cp.cuda.Event(), cp.cuda.Event()
        t0.record()
        for _ in range(20):
            launch()
        t1.record()
        t1.synchronize()
        ms = cp.cuda.get_elapsed_time(t0, t1) / 20
        gbs = N * 4 / ms / 1e6
        print(f"{name:<14}{blocks:>9}{ms:>9.3f}{gbs:>8.0f}{gbs / 3350:>8.0%}")
