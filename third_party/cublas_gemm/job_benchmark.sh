#!/bin/bash
#SBATCH --job-name=cublas_gemm
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --time=0-00:15
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

# e.g. sbatch --export=ALL,CONTAINER="../CuteDSL129.sif" job_benchmark.sh
echo using container $CONTAINER
module load apptainer

apptainer exec --nv $CONTAINER bash -c "
    cd ${SCRIPT_DIR} &&
    make &&
    ./cublas_gemm 65536 4096 8192
"
