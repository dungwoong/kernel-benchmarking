#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --job-name=KernelsProfile
#SBATCH --output=slurm_outputs/KernelsProfile_%j.out
#SBATCH --error=slurm_outputs/KernelsProfile_%j.err

# e.g. sbatch --export=ALL,CONTAINER="../CuteDSL2.sif",SCRIPT="run_scripts/profile_rmsnorm_linear.sh",OUTPUT="profile_data.csv" slurm_profile.sh
echo using container $CONTAINER
echo using script $SCRIPT
module load apptainer

# Create and bind the flashinfer cache to apptainer to not run into disk quota errors
# For all the other scripts can be saved into ~/.bashrc
mkdir -p $SCRATCH/flashinfer_cache
export APPTAINER_BINDPATH="$SCRATCH/flashinfer_cache:$HOME/.cache/flashinfer"

apptainer exec --nv \
  $CONTAINER bash -c "source ./add_path.sh && $SCRIPT > $OUTPUT"
