# Profiles CuteDSL vs Torch gemm for available GEMM shapes
# uses m128n256 as the tile size

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py

while read -r m n k; do
    python3 $DIR/demo/gemm.py "$m" "$n" "$k" --to_csv
    sleep 2
done < $DIR/res/cdsl_gemm_shapes_m128n256.txt