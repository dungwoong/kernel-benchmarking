# Run with (give more CPUs to speed up the autotune):
# sbatch --cpus-per-task=8 --mem=64G --time=03:00:00  --export=ALL,CONTAINER="kernel_bench_apptainer.sif",SCRIPT="run_scripts/autotune_lora.sh" slurm_profile.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CSV="${1:-results/helion_autotune_lora_$(date +%Y%m%d_%H%M%S).csv}"
CONFIGS="${CONFIGS:-$(seq 0 10)}" # indices into GEMM_SHAPES

export HELION_AUTOTUNE_COMPILE_TIMEOUT="${HELION_AUTOTUNE_COMPILE_TIMEOUT:-240}"
export HELION_AUTOTUNE_PRECOMPILE="${HELION_AUTOTUNE_PRECOMPILE:-spawn}"
# Helion uses os.cpu() to set HELION_AUTOTUNE_PRECOMPILE_JOBS default value
# os.cpu() is 64 on slurm machine so use nproc to set the correct default value (specified by --cpus-per-task)
export HELION_AUTOTUNE_PRECOMPILE_JOBS="${HELION_AUTOTUNE_PRECOMPILE_JOBS:-$(nproc)}"

mkdir -p "$(dirname "$CSV")"
echo writing csv to $CSV
python3 "$DIR/demo/gen_header.py" ExperimentOutput > "$CSV"

for i in $CONFIGS; do
    echo "=== autotuning helion lora config $i ==="
    python3 "$DIR/demo/helion/lora_autotune.py" "$i" --csv "$CSV"
done
