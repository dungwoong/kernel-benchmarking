# Setup

Compute Canada cluster (H100, Apptainer).

## Clone

```bash
git clone https://github.com/dungwoong/kernel-benchmarking.git && cd kernel-benchmarking
git submodule update --init --recursive
```

## Build the container (login node)

```bash
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer/cache
export APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

apptainer build --fakeroot CuteDSL2.sif apptainer.def
apptainer run --app setup CuteDSL2.sif
```

## Build FA3 (compute node, GPU)

```bash
salloc --gres=gpu:h100:1 --cpus-per-task=8 --mem=32G --time=03:00:00
apptainer run --nv --app build_fa3 CuteDSL2.sif
```

For the apptainer build_fa3 script optionally can specify MAX_JOBS=8.

## Run a benchmark

```bash
sbatch --export=ALL,CONTAINER=CuteDSL2.sif,SCRIPT=run_scripts/profile.sh,OUTPUT=out.csv slurm_profile.sh
```

`slurm_profile.sh` handles `APPTAINER_BINDPATH` and module loading.

## Optional: persist env so you don't re-export for each shell

```bash
cat >> ~/.bashrc <<'EOF'
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer/cache
export APPTAINER_TMPDIR=${SLURM_TMPDIR:-$SCRATCH/apptainer_tmp}
export APPTAINER_BINDPATH="$SCRATCH/flashinfer_cache:$HOME/.cache/flashinfer"
EOF
```

## Optional: clean rebuild (free disk quota)

```bash
rm -f CuteDSL2.sif
rm -rf .venv
rm -rf $SCRATCH/kernel-benchmarking-venv         # (if venv is symlinked to scratch)
rm -rf $SCRATCH/.apptainer/cache                 # Docker layer blobs (~6 GB, will redownload)
rm -rf $SCRATCH/.cache/pip                       # pip wheel cache (~3 GB, will redownload)
rm -rf $SCRATCH/apptainer_tmp/*
```

Then redo from `Build the container`.
