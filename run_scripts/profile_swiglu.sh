# Output torch vs cublas timings for SwiGLU shapes
# reuse rmsnorm shapes for this

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py ExperimentOutput

while read -r m n k; do
    python3 $DIR/demo/swiglu.py "$m" "$n" "$k" --to_csv
    sleep 3
done < $DIR/res/cdsl_rmsnorm_shapes_m128n256.txt
