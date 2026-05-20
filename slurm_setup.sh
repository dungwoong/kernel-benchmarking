cat >> ~/.bashrc <<'EOF'

# kernel-benchmarking apptainer config
export CONTAINER=kernel_bench_apptainer.sif
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer/cache
export APPTAINER_TMPDIR=${SLURM_TMPDIR:-$SCRATCH/apptainer_tmp}
EOF
source ~/.bashrc
mkdir -p "$APPTAINER_CACHEDIR" "$SCRATCH/apptainer_tmp" "$SCRATCH/flashinfer_cache"

# Downloads for building FA3 on compute node
mkdir -p ~/.flashattn/nvidia/nvcc ~/.flashattn/nvidia/ptxas
curl -fL -o ~/nvcc.tar.xz https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz
tar -xf ~/nvcc.tar.xz -C ~/.flashattn/nvidia/nvcc
curl -fL -o ~/ptxas.tar.xz https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
tar -xf ~/ptxas.tar.xz -C ~/.flashattn/nvidia/ptxas
rm ~/nvcc.tar.xz ~/ptxas.tar.xz