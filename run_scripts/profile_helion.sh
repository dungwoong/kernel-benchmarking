# Profile the Helion kernels against torch
# Comment out a block to skip that workload
# Loads tuned configs from helion_utils/autotune_cache 

# Run as (more memory for attention)
# sbatch --mem=64G --time=01:00:00 --export=ALL,CONTAINER="kernel_bench_apptainer.sif",SCRIPT="run_scripts/profile_helion.sh" slurm_profile.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAMP=$(date +%Y%m%d_%H%M%S)


CSV="results/helion_matmul_$STAMP.csv"
echo writing csv to $CSV
python3 "$DIR/demo/gen_header.py" ExperimentOutput > "$CSV"
for i in $(seq 0 10)
do
    python3 "$DIR/demo/helion/matmul.py" "$i" --csv "$CSV"
done

CSV="results/helion_rmsnorm_linear_$STAMP.csv"
echo writing csv to $CSV
python3 "$DIR/demo/gen_header.py" ExperimentOutput > "$CSV"
for i in $(seq 0 10)
do
    python3 "$DIR/demo/helion/rmsnorm_linear.py" "$i" --csv "$CSV"
done

CSV="results/helion_swiglu_$STAMP.csv"
echo writing csv to $CSV
python3 "$DIR/demo/gen_header.py" ExperimentOutput > "$CSV"
for i in $(seq 0 10)
do
    python3 "$DIR/demo/helion/swiglu.py" "$i" --csv "$CSV"
done

CSV="results/helion_lora_$STAMP.csv"
echo writing csv to $CSV
python3 "$DIR/demo/gen_header.py" ExperimentOutput > "$CSV"
for i in $(seq 0 10)
do
    python3 "$DIR/demo/helion/lora.py" "$i" --csv "$CSV"
done

CSV="results/helion_attention_$STAMP.csv"
echo writing csv to $CSV
python3 "$DIR/demo/gen_header.py" AttentionOutput > "$CSV"
for i in $(seq 0 7)
do
    python3 "$DIR/demo/helion/attention.py" "$i" --csv "$CSV"
done
