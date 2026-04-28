// Same imports as in fa gemm cublas ref
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// For providing stats median/mean/std
#include <algorithm>
#include <numeric>
#include <cmath>

#define CUDA_CHECK(err)   cuda_check(err,   __FILE__, __LINE__) // Expand to current file and line
#define CUBLAS_CHECK(err) cublas_check(err, __FILE__, __LINE__)

static void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void cublas_check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[cuBLAS ERROR] %s:%d: status=%d\n", file, line, (int)status);
        exit(EXIT_FAILURE);
    }
}
// Interface: C[M×N] = alpha * A[M×K] * B[N×K]^T + beta * C[M×N]
// cuBLAS is column-major: it reads row-major A[M×K] as col-major A^T[K×M]. 
// We want C = A*B^T, equivalently (B*A^T)^T in col-major, so pass B first, A second. 
// CUBLAS_OP_T on B: cuBLAS sees B^T[K×N], transpose gives B[N×K].  
// CUBLAS_OP_N on A: cuBLAS sees A^T[K×M], no-op leaves A^T[K×M]. Product: B*A^T = (A*B^T)^T.
// N,M instead of M,N because the col-major result is [N×M] which row-major reads as [M×N]. 
static void run_cublas(cublasHandle_t handle, int M, int N, int K, 
                        float alpha, float beta, 
                        const __nv_bfloat16 *dA, 
                        const __nv_bfloat16 *dB,
                        __nv_bfloat16 *dC) { // Output, not const — we write the result here
    CUBLAS_CHECK(cublasGemmEx(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_N, // Latter is no op
        N, M, K, 
        &alpha, 
        dB, CUDA_R_16BF, K,
        dA, CUDA_R_16BF, K,                                                                                                                                                                                             
        &beta,
        dC, CUDA_R_16BF, N,                                                                                                                                                                                             
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void benchmark_one(cublasHandle_t handle, 
                        int M, int N, int K, 
                        int warmup, int bench, 
                        __nv_bfloat16 *dA, __nv_bfloat16 *dB, __nv_bfloat16 *dC) {
    float alpha = 1.0f, beta = 0.0f; // Fix these
    for (int i = 0; i < warmup; i++)
        run_cublas(handle, M, N, K, alpha, beta, dA, dB, dC);
    // Block CPU until GPU finishes all warmup work before starting the clock.
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start[bench], stop[bench];
    for (int i = 0; i < bench; i++) {
        CUDA_CHECK(cudaEventCreate(&start[i]));
        CUDA_CHECK(cudaEventCreate(&stop[i]));
    }

    std::vector<float> ms(bench);
    for (int i = 0; i < bench; i++) {
        CUDA_CHECK(cudaEventRecord(start[i]));
        run_cublas(handle, M, N, K, alpha, beta, dA, dB, dC);
        CUDA_CHECK(cudaEventRecord(stop[i]));
    }
    // CUDA_CHECK(cudaEventSynchronize(stop[bench-1]));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < bench; i++) {
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], start[i], stop[i]));
        CUDA_CHECK(cudaEventDestroy(start[i]));
        CUDA_CHECK(cudaEventDestroy(stop[i]));
    }

    std::sort(ms.begin(), ms.end());
    float median = ms[bench / 2];
    float mean   = std::accumulate(ms.begin(), ms.end(), 0.0f) / bench;
    float var    = 0.0f;
    for (float t : ms) var += (t - mean) * (t - mean);
    float std_ms = std::sqrt(var / bench);

    // 2*M*N*K: K multiply-adds per output element, M*N output elements, 2 ops per multiply-add.
    double flops  = 2.0 * (double)M * N * K;
    double tflops = flops / (median * 1e-3) / 1e12;

    // printf("M=%-6d N=%-6d K=%-6d | median=%8.4f ms  mean=%8.4f ms  std=%7.4f ms  |  %7.2f TFLOPS\n",
    //        M, N, K, median, mean, std_ms, tflops);
    printf("%d,%d,%d,%.6f,%.6f,%.6f,%.2f\n", M, N, K, median, mean, std_ms, tflops);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    int warmup = 10, bench = 50;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s M N K [warmup [bench]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    if (argc >= 5) warmup = atoi(argv[4]);
    if (argc >= 6) bench  = atoi(argv[5]);

    CUDA_CHECK(cudaSetDevice(0));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Allocate device memory
    __nv_bfloat16 *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)N * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(__nv_bfloat16)));

    // Initialize A and B on host with random values in [-1, 1], then copy to device.
    std::vector<__nv_bfloat16> hA((size_t)M * K);
    std::vector<__nv_bfloat16> hB((size_t)N * K);
    srand(42); // Fix seed for reproducibility across runs
    for (auto &x : hA) x = __float2bfloat16((rand() % 200 - 100) * 0.01f);
    for (auto &x : hB) x = __float2bfloat16((rand() % 200 - 100) * 0.01f);
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // printf("cuBLAS bf16 GEMM  (k-major A and B, C = A * B^T)\n");
    // printf("warmup=%d  bench=%d\n\n", warmup, bench);

    benchmark_one(handle, M, N, K, warmup, bench, dA, dB, dC);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}