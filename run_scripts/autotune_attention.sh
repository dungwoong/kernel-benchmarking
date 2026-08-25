# specify --cpus-per-task=8 in sbatch command to speed up the autotune

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default CSV with timestamp, specify CSV to overwrite it
CSV="${1:-results/helion_attn_autotune_$(date +%Y%m%d_%H%M%S).csv}"

export HELION_AUTOTUNE_COMPILE_TIMEOUT="${HELION_AUTOTUNE_COMPILE_TIMEOUT:-600}"
export HELION_AUTOTUNE_PRECOMPILE="${HELION_AUTOTUNE_PRECOMPILE:-spawn}"
# Helion uses os.cpu() to set HELION_AUTOTUNE_PRECOMPILE_JOBS default value
# os.cpu() is 64 on slurm machine so use nproc to set the correct default value (specified by --cpus-per-task)
export HELION_AUTOTUNE_PRECOMPILE_JOBS="${HELION_AUTOTUNE_PRECOMPILE_JOBS:-$(nproc)}"

mkdir -p "$(dirname "$CSV")"
echo writing csv to $CSV
python3 "$DIR/demo/gen_header.py" AttentionOutput > "$CSV"

for i in $(seq 0 7)
do
    python3 "$DIR/demo/helion/attention_autotune.py" "$i" --csv "$CSV"
done
