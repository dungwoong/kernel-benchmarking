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

# Intermediate hook

```
import ttgir_hook
ttgir_hook.install(cut="tritongpu-coalesce", kernel="_helion_rmsnorm_lin_kernel", workdir="kernel_bundles/ttgir_hook")

# run this, will create .orig.mlir and .final.ttgir
TRITON_ALWAYS_COMPILE=1 python3 demo/helion/rmsnorm_linear.py 1 2>&1 | tee run.log
```

Run that, try making a .edit.mlir and run again. It will use the edit so you gotta see what happens.