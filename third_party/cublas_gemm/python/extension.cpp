#include <torch/extension.h>
#include <cublas_v2.h>

#define CHECK_CUDA_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA_DEVICE(x); CHECK_CONTIGUOUS(x);

torch::Tensor gemm_cuda(
    int m, int n, int k,
    at::BFloat16 *dA, at::BFloat16 *dB
);

void cublas_init_cuda();

void init_cublas() {
    cublas_init_cuda();
}

// ABt
torch::Tensor hgemm(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto m = A.sizes()[0];
    auto n = B.sizes()[0];
    auto k = A.sizes()[1];

    auto result = gemm_cuda(
        m, n, k, A.data_ptr<at::BFloat16>(), B.data_ptr<at::BFloat16>()
    );
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_cublas", &init_cublas, "Initialize cuBLAS");
    m.def("cublas_hgemm", &hgemm, "cuBLAS hgemm");
}