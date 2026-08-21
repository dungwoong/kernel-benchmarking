# Profiles compiler_2 vs CuteDSL vs Torch vs TRT lora over the GEMM_SHAPES configs

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default CSV file, pass a path as the first argument to override
CSV="${1:-results/lora_$(date +%Y%m%d_%H%M%S).csv}"

# Write the header into CSV, overwrites the file
mkdir -p "$(dirname "$CSV")"
echo writing csv to $CSV
python3 $DIR/demo/gen_header.py ExperimentOutput > "$CSV"

for i in $(seq 0 10)
do
    python3 $DIR/demo/new_profiling/lora_new.py $i --csv "$CSV"
    sleep 3
done