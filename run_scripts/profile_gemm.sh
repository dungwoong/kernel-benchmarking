DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py ExperimentOutput

for i in $(seq 0 10) 
do
    python3 $DIR/demo/new_profiling/gemm_new.py $i
    sleep 3
done