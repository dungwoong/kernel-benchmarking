#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --job-name=KernelsProfile
#SBATCH --output=slurm_outputs/KernelsProfile_%j.out
#SBATCH --error=slurm_outputs/KernelsProfile_%j.err

# e.g. sbatch --export=ALL,CONTAINER="kernel_bench_apptainer.sif",SCRIPT="run_scripts/profile_rmsnorm_linear.sh",OUTPUT="profile_data.csv" slurm_profile.sh
echo using container $CONTAINER
echo using script $SCRIPT
module load apptainer

apptainer exec --nv \
  $CONTAINER bash -c "source ./add_path.sh && $SCRIPT > $OUTPUT"
