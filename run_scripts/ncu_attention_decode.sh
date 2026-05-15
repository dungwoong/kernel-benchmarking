# NCU profiles for each (q_len, kv_len, nheads) shape
# one .ncu-rep file per shape, all 6 baselines captured inside

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"
OUT_DIR="$DIR/ncu_reports"
mkdir -p "$OUT_DIR"

while read -r q_len kv_len nheads; do
    REPORT="$OUT_DIR/attention_decode_q${q_len}_kv${kv_len}_h${nheads}"
    echo "ncu -> $REPORT"
    ncu --set full -f -o "$REPORT" \
        python3 $DIR/demo/attention_decode_ncu.py "$q_len" "$kv_len" "$nheads"
done < $DIR/res/attention_decode_shapes.txt
