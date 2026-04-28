# Profile torch gemm

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py

while read -r m n k; do
    python3 $DIR/demo/gemm_torch.py "$m" "$n" "$k" --to_csv
    sleep 3
done < $DIR/res/gemm_shapes.txt
