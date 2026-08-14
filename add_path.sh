REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO_ROOT/.venv/bin/activate" ] && source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/python:$REPO_ROOT/third_party/fused_kernels_cutedsl_lib/python:$REPO_ROOT/third_party/cublas_gemm/python:$REPO_ROOT/third_party/helion_kernels_lib/python:$REPO_ROOT/third_party/compiler_2/python"
export TORCH_CUDA_ARCH_LIST="9.0" # for torch extensions
export C2_CACHE_DIR="$REPO_ROOT/.cache/c2" 
export CUDA_HOME=/usr/local/cuda # override host module-system path to container's CUDA 12.9
unset REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_FILE PIP_CERT # host CA paths don't exist in Ubuntu container