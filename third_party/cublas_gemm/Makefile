NVCC    = nvcc
ARCH    = sm_90        # H100 is sm_90
CFLAGS  = -O3 -arch=$(ARCH) -std=c++17
LIBS    = -lcublas

TARGET  = cublas_gemm
SRC     = cublass_gemm.cu

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)
