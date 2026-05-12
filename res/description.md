### cdsl_rmsnorm_shapes
- Contains some shapes that could conceivably be found in an LLM e.g. swiglu up-projection, down-projection, etc.
- e.g. in LLaMA you have seqlen and e.g. 32 heads, 128 dim = 4096 `d_model`, for swiglu + up-projection that would be 32768 in total as the up-projection
- Contains some vanity shapes that just show speedups