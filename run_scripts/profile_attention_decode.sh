# torch SDPA + FlashInfer + FA2 + FA3 timings 

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default CSV file, pass a path as the first argument to override
CSV="${1:-results/attention_decode_$(date +%Y%m%d_%H%M%S).csv}"

# Write the header into CSV, overwrites the file
mkdir -p "$(dirname "$CSV")"
echo writing csv to $CSV
echo "label,q_len,kv_len,nheads,ms_median,ms_mean,ms_std,max_abs,max_rel,rmse,baseline_speedup,metadata,autotune_seconds" > "$CSV"

while read -r q_len kv_len nheads; do
    python3 $DIR/demo/attention_decode.py "$q_len" "$kv_len" "$nheads" --csv $CSV
    sleep 3
done < $DIR/res/attention_decode_shapes.txt