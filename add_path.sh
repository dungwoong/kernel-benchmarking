REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO_ROOT/.venv/bin/activate" ] && source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/python:$REPO_ROOT/third_party/fused_kernels_cutedsl_lib/python:$REPO_ROOT/third_party/cublas_gemm/python"
export TORCH_CUDA_ARCH_LIST="9.0" # for torch extensions