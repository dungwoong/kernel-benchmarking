# Profiles compiler_2 vs CuteDSL vs Torch vs TRT lora over the GEMM_SHAPES configs

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

# Default CSV file, comment out to specify where to redirect stdout for unstructured output
CSV="${1:-results/lora_$(date +%Y%m%d_%H%M%S).csv}"

# Write the header into CSV, overwrites the files
# or to stdout when CSV is not set
if [ -n "$CSV" ]; then
    mkdir -p "$(dirname "$CSV")"
    echo writing csv to ${CSV:-"the run script default"}
    python3 $DIR/demo/gen_header.py ExperimentOutput > "$CSV"
else
    python3 $DIR/demo/gen_header.py ExperimentOutput
fi

for i in $(seq 0 10)
do
    python3 $DIR/demo/new_profiling/lora_new.py $i ${CSV:+--csv "$CSV"}
    sleep 3
done