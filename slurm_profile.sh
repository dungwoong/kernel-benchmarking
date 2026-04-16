#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --job-name=KernelsProfile
#SBATCH --output=slurm_outputs/KernelsProfile_%j.out
#SBATCH --error=slurm_outputs/KernelsProfile_%j.err


# e.g. sbatch --export=ALL,CONTAINER="../CuteDSL129.sif" slurm_profile.sh
echo using container $CONTAINER
module load apptainer
apptainer exec --nv $CONTAINER bash -c "source ./add_path.sh && ./run_scripts/profile.sh > profile_data.csv"