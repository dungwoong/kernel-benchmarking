#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --job-name=KernelsProfile
#SBATCH --output=slurm_outputs/KernelsProfile_%j.out
#SBATCH --error=slurm_outputs/KernelsProfile_%j.err

# e.g. sbatch --export=ALL,CONTAINER="kernel_bench_apptainer.sif",SCRIPT="run_scripts/profile_gemm.sh",CSV="results/gemm.csv" slurm_profile.sh
# CSV is passed to the run script as its first argument, leave it unset to use the run script's default
# OUTPUT is an optional non-csv output. In case CSV is specified it will take the rest of the stdout
# leave it unset to let stdout go to the default slurm .out file
echo using container $CONTAINER
echo using script $SCRIPT
echo writing csv to ${CSV:-"the run script default"}
module load apptainer

# Redirect conditional on OUTPUT specified
apptainer exec --nv \
  $CONTAINER bash -c "source ./add_path.sh && $SCRIPT $CSV ${OUTPUT:+> $OUTPUT}"
