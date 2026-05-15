# torch SDPA + FlashInfer + FA2 + FA3 timings 

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

# For the time this header, nonconforming to attention based ExperimentOutput
echo "label,q_len,kv_len,nheads,ms_median,ms_mean,ms_std,max_abs,max_rel,rmse,metadata"

while read -r q_len kv_len nheads; do
    python3 $DIR/demo/attention_decode.py "$q_len" "$kv_len" "$nheads" --to_csv
    sleep 3
done < $DIR/res/attention_decode_shapes.txt
