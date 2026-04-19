Torch kernels can be used as references with fp64 inputs, but also as comparisons with bf16 inputs.

User must run torch.compile separately. This allows us to test eager and compiled.

# Workloads

### Linear Attention

$$o = ((QK)/\sqrt{d} * M)V$$

where M is a lower triangular mask.

### Inference Attention
The kernel must perform matmul to get qkv, update the kv cache, and then get the outputs for the new q tokens.

- $q = xW_q, k = xW_k, v = xW_v$
- add k and v to the k_cache and v_cache
- $o = attn(q, k_{cache}, v_{cache})$

### nTrans
$y = Norm(X + \alpha(Norm(H - X)))$

- Copied from mirage's norm [transformer example](https://github.com/mirage-project/mirage/blob/main/benchmark/baselines/pytorch/norm_transformer.py)
- However, they have an error that I corrected where they reused X instead of (H-X)
- This implementation is close to what they do in the actual nTransformer code ([line 407](https://github.com/lucidrains/nGPT-pytorch/blob/main/nGPT_pytorch/nTransformer.py))
- Alpha is required here as the interpolation variable