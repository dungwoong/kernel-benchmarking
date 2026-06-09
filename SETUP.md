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
apptainer build --fakeroot kernel_bench_apptainer.sif apptainer.def
```

## Build FA3

Then allocate a GPU node and build:

```bash
apptainer run --nv --app build_fa3 kernel_bench_apptainer.sif
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

**Slurm Setup**

To set up the environment variables for apptainer and download dependencies for FA3 build:

```bash
./slurm_setup.sh
```

**Build the apptainer**

Same as in the regular instructions. Prepend with `env -u APPTAINER_BINDPATH` to unset the environment variable if it has been set before.

**Build FA3**

Allocate a GPU node and build:

```bash
salloc --gres=gpu:h100:1 --cpus-per-task=8 --mem=32G --time=03:00:00
apptainer run --nv --app build_fa3 kernel_bench_apptainer.sif
```

**Run scripts**

After the apptainer build add `export APPTAINER_BINDPATH=$SCRATCH/flashinfer_cache:$HOME/.cache/flashinfer` to the ~/.bashrc to avoid flashinfer cache exceeding the home disk quota. The apptainer inherits this from the host via sbatch --export=ALL.

And create the folder `mkdir -p $SCRATCH/flashinfer_cache`.

Then run `sbatch --export=ALL,SCRIPT="run_scripts/profile.sh",OUTPUT="out.csv" slurm_profile.sh`

**Clean rebuild** to free disk quota:

```bash
rm -f kernel_bench_apptainer.sif
rm -rf .venv
rm -rf $SCRATCH/.apptainer/cache $SCRATCH/apptainer_tmp/* $SCRATCH/flashinfer_cache/*
```

## Install Additional Libraries
This will install packages into a virtual environment. We have some lightweight packages that may update periodically, which are useful to install outside of the .sif build process.

Run this from the root of the folder.
```
apptainer exec <CONTAINER_PATH> bash -c 'source ./add_path.sh && pip install -r venv_requirements.txt'
```