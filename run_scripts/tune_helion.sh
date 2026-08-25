DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. & pwd)"

for i in $(seq 0 10) # number of shapes
do
    for f in matmul.py rmsnorm_linear.py swiglu.py
    do
        python3 $DIR/demo/helion/$f $i
    done
done