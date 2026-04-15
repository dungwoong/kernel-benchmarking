DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py

for VAL in 1024 2048 4096 8192 16384 32768
do
    python3 $DIR/demo/gemm.py $VAL $VAL $VAL --to_csv
    python3 $DIR/demo/lora.py $VAL $VAL $VAL 16 --to_csv
    python3 $DIR/demo/rmsnorm_linear.py $VAL $VAL $VAL --to_csv
    python3 $DIR/demo/swiglu.py $VAL $VAL $VAL --to_csv
    sleep 2
done