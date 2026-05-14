# torch SDPA + FlashInfer + FA2 + FA3 timings 

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py ExperimentOutput

while read -r m n k; do
    python3 $DIR/demo/attention_decode.py "$m" "$n" "$k" --to_csv
    sleep 3
done < $DIR/res/attention_decode_shapes.txt
