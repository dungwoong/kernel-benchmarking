#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>

#define CUDA_CHECK(err)   cuda_check(err,   __FILE__, __LINE__) // Expand to current file and line
#define CUBLAS_CHECK(err) cublas_check(err, __FILE__, __LINE__)

static void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", file, line, cudaGetErrorString(err));
        throw std::runtime_error("cuda error");
    }
}

static void cublas_check(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[cuBLAS ERROR] %s:%d: status=%d\n", file, line, (int)status);
        throw std::runtime_error("cublas error");
    }
}

cublasHandle_t handle;

void cublas_init_cuda() {
    CUBLAS_CHECK(cublasCreate(&handle));
}

torch::Tensor gemm_cuda (
    int m, int n, int k, at::BFloat16 *dA, at::BFloat16 *dB
) {
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    torch::Tensor C = torch::zeros({m, n}, options);
    auto dC = C.data_ptr<at::BFloat16>();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_N, // Latter is no op
        n, m, k, 
        &alpha, 
        (__nv_bfloat16 *)dB, CUDA_R_16BF, k,
        (__nv_bfloat16 *)dA, CUDA_R_16BF, k,                                                                                                                                                                                             
        &beta,
        (__nv_bfloat16 *)dC, CUDA_R_16BF, n,                                                                                                                                                                                             
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    return C;
}