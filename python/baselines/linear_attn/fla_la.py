from fla.ops.simple_gla.chunk import chunk_simple_gla_fwd

# this version links to triton kernels directly
# we only support qkv and scale though
# this returns 2 things, o and ht
fla_linear_attention = chunk_simple_gla_fwd