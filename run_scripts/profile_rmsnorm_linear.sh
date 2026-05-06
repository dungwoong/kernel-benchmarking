# Output torch vs cublas timings for RMSNorm+Linear shapes

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py ExperimentOutput

while read -r m n k; do
    python3 $DIR/demo/rmsnorm_linear.py "$m" "$n" "$k" --to_csv
    sleep 3
done < $DIR/res/cdsl_rmsnorm_shapes_m128n256.txt
