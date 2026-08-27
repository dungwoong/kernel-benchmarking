DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

# Default CSV file, pass a path as the first argument to override
CSV="${1:-results/profile_new_$(date +%Y%m%d_%H%M%S).csv}"

# Write the header into CSV, overwrites the file
mkdir -p "$(dirname "$CSV")"
echo writing csv to $CSV
python3 $DIR/demo/gen_header.py ExperimentOutput > "$CSV"

for i in $(seq 0 10) # number of shapes
do
    for f in gemm_new.py rmsnorm_linear_new.py swiglu_new.py lora_new.py
    do
        python3 $DIR/demo/new_profiling/$f $i --csv "$CSV"
        sleep 3
    done
done