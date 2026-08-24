import math
import time
import torch
import flashinfer
import flash_attn_interface
from profile_utils import ExperimentOutput, get_normal_bernoulli, get_attention_args, write_csv_rows
from cutedsl_kernels import DAttn2, DAttnSplit1, AttnReduce1
from cdsl_helpers.cdsl_fn_utils import compile_cutedsl
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernels.hel.attention import get_kernel as get_hel_kernel
from compiler import compile_hel

torch.manual_seed(18)

BATCH = 1
HEAD_DIM = 128
CAUSAL = False

# hel attention config: tile_m must be a multiple of 64, kv_len a multiple of tile_n
HEL_TILE_M, HEL_TILE_N, HEL_STAGES = 64, 128, 2


def naive_attention(q, k_cache, v_cache):
    # q, k_cache, v_cache are (B, S, H, D), returns (B, Sq, H, D)
    qh = q.transpose(1, 2)
    kh = k_cache.transpose(1, 2)
    vh = v_cache.transpose(1, 2)
    scale = 1.0 / math.sqrt(qh.size(-1))
    scores = (qh @ kh.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.transpose(1, 2).contiguous()


class AttentionBaseline:
    """holds a contiguous KV cache (kv_len, H, D) pre-populated to kv_len
    each __call__ times only the attention op
    """

    def __init__(self, cache_K, cache_V):
        self.cache_K = cache_K
        self.cache_V = cache_V


@torch.compile
def inductor_attention(q, k, v):
    # all head-major: (H, q_len, D) x (H, kv_len, D) -> (H, q_len, D)
    scale = (HEAD_DIM ** -0.5)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)

    # (nheads, q_len, kv_len) x (nheads, kv_len, dim) -> (nheads, q_len, dim)
    output = torch.matmul(attn_probs, v)
    return output


class InductorBaseline(AttentionBaseline):
    def __init__(self, cache_K, cache_V):
        # the matmuls want head-major, so flip the cache to (H, S, D) once here
        super().__init__(cache_K.transpose(0, 1), cache_V.transpose(0, 1))

    def __call__(self, q):
        # (B, Sq, H, D) -> (H, Sq, D), a free view
        qh = q.squeeze(0).transpose(0, 1)
        # back to (Sq, H, D) so it lines up with the reference
        return inductor_attention(qh, self.cache_K, self.cache_V).transpose(0, 1)


class FlashInferBaseline(AttentionBaseline):
    def __call__(self, q):
        out = flashinfer.single_prefill_with_kv_cache(
            q=q.squeeze(0),
            k=self.cache_K,
            v=self.cache_V,
            kv_layout="NHD",
            causal=CAUSAL,
        )
        # This does not work since flashinfer expects 1 new token ONLY, but this is decoding attention
        # out = flashinfer.single_decode_with_kv_cache(
        #     q = q.squeeze(0).squeeze(0),
        #     k = self.cache_K,
        #     v = self.cache_V,
        #     kv_layout="NHD",
        #     # causal=CAUSAL,
        # )
        return out.unsqueeze(0)


class FA3Baseline(AttentionBaseline):
    def __call__(self, q):
        full_K = self.cache_K.unsqueeze(0)
        full_V = self.cache_V.unsqueeze(0)
        return flash_attn_interface.flash_attn_func(q, full_K, full_V, causal=CAUSAL)


class FA3FlashDecodeBaseline(AttentionBaseline):
    def __call__(self, q):
        return flash_attn_interface.flash_attn_with_kvcache(
            q=q,
            k_cache=self.cache_K.unsqueeze(0),
            v_cache=self.cache_V.unsqueeze(0),
            # cache_seqlens=self.cache_K.shape[0],
            causal=CAUSAL,
        )


class TorchSDPABaseline(AttentionBaseline):
    def __call__(self, q):
        # SDPA wants (B, H, S, D)
        full_K = self.cache_K.unsqueeze(0).transpose(1, 2)
        full_V = self.cache_V.unsqueeze(0).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), full_K, full_V, is_causal=CAUSAL,
        )
        # transpose back to (B, S, H, D)
        return o.transpose(1, 2)


dattn = DAttn2(
        qk_mnk=(16, 128, 128),
        stages=2,
        p_stages=1,
        is_persistent=False
    )

NSPLITS = 4
dattn_split = DAttnSplit1(
    qk_mnk=(16, 128, 128),
    stages=2,
    p_stages=1,
    k_splits=NSPLITS,
    )
dattn_reduce = AttnReduce1(n=128, splits=NSPLITS)

class CDSLAttention(AttentionBaseline):
    def with_kernel(self, kernel):
        self.kernel = kernel
        return self
    
    def __call__(self, q):
        q = q.squeeze(0)
        o = torch.empty_like(q)
        multiplier = q.shape[-1] ** -0.5
        self.kernel(q, self.cache_K, self.cache_V, o, multiplier)
        return o

class CDSLAttentionSplitK(AttentionBaseline):
    def with_kernel(self, kernel, reduction_kernel):
        self.kernel = kernel
        self.reduce = reduction_kernel
        return self
    
    def __call__(self, q):
        q = q.squeeze(0)
        M, H, D = q.shape
        multiplier = D ** -0.5
        o = torch.empty((H, M, NSPLITS, D), dtype=torch.float32, device='cuda')
        osum = torch.empty((H, NSPLITS, M), dtype=torch.float32, device='cuda')
        ofinal = torch.empty((M, H, D), dtype=torch.bfloat16, device='cuda')
        self.kernel(q, self.cache_K, self.cache_V, o, osum, multiplier)
        self.reduce(o, osum, ofinal)
        return ofinal

class HelAttention(AttentionBaseline):
    """hel attention: q/k/v/o are head-major (H, S, D), cache is stored that way"""
    def with_kernel(self, kernel):
        self.kernel = kernel
        return self

    def __call__(self, q):
        # (B, Sq, H, D) -> (H, Sq, D), a free view since the kernel reads strides from the tensor
        qh = q.squeeze(0).transpose(0, 1)
        H, M, D = qh.shape
        o = torch.empty((H, M, D), dtype=torch.bfloat16, device='cuda')
        self.kernel(qh, self.cache_K, self.cache_V, o)
        # back to (Sq, H, D) so it lines up with the reference
        return o.transpose(0, 1)


if __name__ == "__main__":
    args = get_attention_args()
    q_len, kv_len, nheads = args.q_len, args.kv_len, args.nheads

    # KV cache pre-populated with kv_len tokens
    full_K_64 = get_normal_bernoulli(
        (kv_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )
    full_V_64 = get_normal_bernoulli(
        (kv_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )

    q_64 = get_normal_bernoulli(
        (BATCH, q_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )

    # kernels run in bf16
    full_K = full_K_64.to(torch.bfloat16)
    full_V = full_V_64.to(torch.bfloat16)
    q = q_64.to(torch.bfloat16)

    ref = naive_attention(
        q_64,
        full_K_64.unsqueeze(0),
        full_V_64.unsqueeze(0),
    )

    # Compile CDSL kernels
    # can reuse q as o for compile purposes
    multiplier = HEAD_DIM ** -0.5
    q_ = q.squeeze(0)
    cdsl_tensors = (q_, full_K, full_V, q_, multiplier)
    compiled_attn = compile_cutedsl(cdsl_tensors, dattn, False)

    O_split = torch.empty((nheads, q_len, NSPLITS, HEAD_DIM), dtype=torch.float32, device='cuda')
    Osum_split = torch.empty((nheads, NSPLITS, q_len), dtype=torch.float32, device='cuda')
    compiled_attn_splitk = compile_cutedsl((q_, full_K, full_V, O_split, Osum_split, multiplier), dattn_split, False)
    compiled_attn_reduce = compile_cutedsl((O_split, Osum_split, q_), dattn_reduce, False)

    # each baseline gets its own cache copy
    def make_cache():
        return full_K.clone(), full_V.clone()

    # hel wants head-major (H, S, D) tensors, and one m-tile covering all of q_len
    def make_hel_cache():
        return full_K.clone().transpose(0, 1), full_V.clone().transpose(0, 1)

    assert kv_len % HEL_TILE_N == 0, f'hel kernel needs kv_len divisible by {HEL_TILE_N}'
    hel_q_len = ((q_len + HEL_TILE_M - 1) // HEL_TILE_M) * HEL_TILE_M
    hel_K, hel_V = make_hel_cache()
    hel_O = torch.empty((nheads, q_len, HEAD_DIM), dtype=torch.bfloat16, device='cuda')
    compiled_attn_hel = compile_hel(
        get_hel_kernel(hel_q_len, kv_len, HEAD_DIM, nheads, HEL_TILE_M, HEL_TILE_N, HEL_STAGES, q_stages=1),
        q_.transpose(0, 1), hel_K, hel_V, hel_O,
        name=f'hel_attention_{q_len}x{kv_len}x{nheads}',
    )

    pairs = []
    for name, cls in [
        ("attn_torch_sdpa",      TorchSDPABaseline),
        ("attn_inductor",        InductorBaseline),
        ("attn_flashinfer",      FlashInferBaseline),
        ("attn_fa3",             FA3Baseline),
        ("attn_fa3_flashdecode", FA3FlashDecodeBaseline),
        ("attn_cdsl",            CDSLAttention),
        ('attn_cdsl_splitk',     CDSLAttentionSplitK),
        ('attn_hel',             HelAttention),
    ]:
        cK, cV = make_hel_cache() if name == 'attn_hel' else make_cache()
        baseline = cls(cache_K=cK, cache_V=cV)
        if name == 'attn_cdsl':
            baseline = baseline.with_kernel(compiled_attn)
        elif name == 'attn_cdsl_splitk':
            baseline = baseline.with_kernel(compiled_attn_splitk, compiled_attn_reduce)
        elif name == 'attn_hel':
            baseline = baseline.with_kernel(compiled_attn_hel)
        pairs.append((name, baseline))

    
    # CuDNN seems to work fine in the decoding case, so we can make torch sdpa use cudnn
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        tensors = (q,)
        outputs = []
        for name, baseline in pairs:
            out = ExperimentOutput(name, q_len, kv_len, nheads)
            out.run(baseline, tensors, ref)
            outputs.append(out)
            time.sleep(2)

    if args.csv is not None:
        # write_csv_rows appends to args.csv, or prints the rows when it's None
        write_csv_rows([ExperimentOutput.list_to_csv(out.values()) for out in outputs], args.csv)
    else:
        print(ExperimentOutput.header())
        for out in outputs:
            print(out.values())
