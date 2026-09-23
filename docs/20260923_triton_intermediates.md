```
# Inside helion running file do this
import triton_bundle
triton_bundle.install("kernel_bundles", only="_helion_rmsnorm_lin_kernel")

this will put all of the relevant intermediate AND json files into a folder so you can modify and recompile starting there.

# Recompiles ttgir or ttir(only tested ttgir) from dump
python3 demo/compile_from_ir.py ./kernel_bundles/_helion_rmsnorm_lin_kernel/1238b367a7fb/_helion_rmsnorm_lin_kernel
```

You can diffcheck the ttir etc, modify and re-run.
Currently compile_from_ir is hardcoded to 4096 input sizes.