// Kernel 1 from the Harris slides (interleaved addressing), with a host program
// that checks the answer and measures bandwidth. Copy this file to try your own version.
//
// Run:  modal run kernels/run_cu.py --path kernels/w1/reduce1.cu
#include <cstdio>
#include <vector>

__global__ void reduce1(const float *g_idata, float *g_odata) {
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

int main() {
    const int N = 1 << 26;            // 67M floats = 268 MB
    const int BLOCK = 128;            // threads per block
    const int BLOCKS = N / BLOCK;     // one partial sum per block

    std::vector<float> h_in(N, 1.0f); // all ones, so the right answer is exactly N
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, BLOCKS * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    reduce1<<<BLOCKS, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out);  // warm-up

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < 20; r++)
        reduce1<<<BLOCKS, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 20;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<float> h_out(BLOCKS);
    cudaMemcpy(h_out.data(), d_out, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (float p : h_out) sum += p;   // add the partial sums on the CPU

    // Peak bandwidth of this GPU: 2 transfers per clock (DDR) x clock x bus width in bytes.
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    double peak = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9;

    double gbs = (double)N * sizeof(float) / ms / 1e6;
    printf("%s, peak %.0f GB/s\n", p.name, peak);
    printf("sum %.0f, want %d: %s\n", sum, N, sum == N ? "OK" : "WRONG");
    printf("%.3f ms  %.0f GB/s  %.1f%% of peak\n", ms, gbs, gbs / peak * 100);
    return 0;
}
