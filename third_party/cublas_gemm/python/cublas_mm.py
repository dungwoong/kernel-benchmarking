from torch.utils.cpp_extension import load
import os
import sys
import contextlib

current_file = os.path.abspath(__file__)
base_path = os.path.dirname(current_file)

if not os.path.exists(base_path + "/build"):
    os.makedirs(base_path + "/build")

with contextlib.redirect_stdout(sys.stderr): # in case we want to redirect stdout to a file, don't include compilation results
    cublas_backend = load(
        name="gemm",
        sources=[os.path.join(base_path, "extension.cpp"), os.path.join(base_path, "ext_kernel.cu")],
        extra_cflags=['-lcuda', '-ldl', '-lcublas'],
        extra_cuda_cflags=[
            '-lcuda', '-ldl', '-lcublas'
        ],
        extra_ldflags=[
            '-lcuda', '-ldl', '-lcublas'
        ],
        build_directory=os.path.join(base_path, 'build'),
        with_cuda=True,
        verbose=True
    )

init_res = cublas_backend.init_cublas()
matmul = cublas_backend.cublas_hgemm