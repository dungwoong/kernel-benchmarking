# Prefill attention compares cuDNN vs FA3 vs Torch over ATTENTION_SHAPES

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default CSV file, pass a path as the first argument to override
CSV="${1:-results/attention_prefill_$(date +%Y%m%d_%H%M%S).csv}"
CONFIGS="${CONFIGS:-$(seq 0 5)}" 

# Write the header into CSV, overwrites the file
mkdir -p "$(dirname "$CSV")"
echo writing csv to $CSV
python3 $DIR/demo/gen_header.py AttentionOutput > "$CSV"

for i in $CONFIGS
do
    python3 $DIR/demo/new_profiling/attention_prefill_new.py $i --csv "$CSV"
    sleep 3
done