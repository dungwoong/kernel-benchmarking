echo m,n,k,med_ms,mean_ms,std_ms,tflops
while read -r m n k; do
    ./cublas_gemm "$m" "$n" "$k"
done < shapes.txt
