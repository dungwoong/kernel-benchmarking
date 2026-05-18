# Setup

Build and run the benchmark container with Apptainer. If you are on a Slurm HPC cluster, read [Slurm HPC clusters](#slurm-hpc-clusters) first.

## Clone

```bash
git clone https://github.com/dungwoong/kernel-benchmarking.git && cd kernel-benchmarking
git submodule update --init --recursive
```

## Build the container

`requirements.txt` is installed into the image at build time, so the `.sif` is self-contained. The build needs internet access.

```bash
apptainer build --fakeroot CuteDSL2.sif apptainer.def
```

## Build FA3

FA3's build downloads nvcc/ptxas toolchain for CUDA 12.9, and compute nodes have no internet. Pre-fetch it on the login node first. It caches to `~/.flashattn`, which the build reuses:

```bash
mkdir -p ~/.flashattn/nvidia/nvcc ~/.flashattn/nvidia/ptxas
curl -fL -o /tmp/nvcc.tar.xz \
  https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz
tar -xf /tmp/nvcc.tar.xz -C ~/.flashattn/nvidia/nvcc
curl -fL -o /tmp/ptxas.tar.xz \
  https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
tar -xf /tmp/ptxas.tar.xz -C ~/.flashattn/nvidia/ptxas
```

Then allocate a GPU node and build:

```bash
apptainer run --nv --app build_fa3 CuteDSL2.sif
```

Optionally pass `MAX_JOBS=8` to parallelize the compile.

**Run benchmarks**

```bash
sbatch --export=ALL,SCRIPT=run_scripts/profile.sh,OUTPUT=out.csv slurm_profile.sh
```

## Slurm HPC clusters

**Apptainer module**

```bash
module load apptainer
```

**Cache variables**

```bash
cat >> ~/.bashrc <<'EOF'

# kernel-benchmarking apptainer config
export CONTAINER=CuteDSL2.sif
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer/cache
export APPTAINER_TMPDIR=${SLURM_TMPDIR:-$SCRATCH/apptainer_tmp}
EOF
source ~/.bashrc
mkdir -p "$APPTAINER_CACHEDIR" "$SCRATCH/apptainer_tmp" "$SCRATCH/flashinfer_cache"
```

**Build FA3**

Build the container, then allocate a GPU node for `build_fa3`:

```bash
salloc --gres=gpu:h100:1 --cpus-per-task=8 --mem=32G --time=03:00:00
apptainer run --nv --app build_fa3 CuteDSL2.sif
```

**Run scripts**

After the build add `export APPTAINER_BINDPATH=$SCRATCH/flashinfer_cache:$HOME/.cache/flashinfer` to the ~/.bashrc to avoid flashinfer cache exceeding the home disk quota. The apptainer inherits this from the host via sbatch --export=ALL.

And create the folder `mkdir -p $SCRATCH/flashinfer_cache`.

Then run `sbatch --export=ALL",SCRIPT="run_scripts/profile.sh",OUTPUT="out.csv" slurm_profile.sh`

**Clean rebuild** to free disk quota:

```bash
rm -f CuteDSL2.sif
rm -rf .venv
rm -rf $SCRATCH/.apptainer/cache $SCRATCH/apptainer_tmp/* $SCRATCH/flashinfer_cache/*
```
