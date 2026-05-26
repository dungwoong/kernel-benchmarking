DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

python3 $DIR/demo/gen_header.py ExperimentOutput

for i in $(seq 0 10) # number of shapes
do
    for f in rmsnorm_linear_new.py swiglu_new.py lora_new.py
    do
        python3 $DIR/demo/new_profiling/$f $i
        sleep 3
    done
done