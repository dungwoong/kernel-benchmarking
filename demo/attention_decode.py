import math
import torch
import flashinfer
import flash_attn
import flash_attn_interface

from profile_utils import ExperimentOutput, get_normal_bernoulli, get_args

torch.manual_seed(18)

BATCH = 1
HEAD_DIM = 128
CAUSAL = False


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
    """pre-allocated contiguous KV cache of shape (kv_len, H, D)
    each __call__ writes the new M tokens into cache_K/cache_V[P:P+M]
    then runs attention against the full cache buffer
    """

    def __init__(self, M, P, H, D, cache_K, cache_V):
        self.M = M
        self.P = P
        self.H = H
        self.D = D
        self.cache_K = cache_K
        self.cache_V = cache_V

    def write_new_kv(self, k_new, v_new):
        # k_new, v_new arrive as (1, M, H, D) from the shared kernel inputs
        # strip the batch dim to match the 3D cache layout, then slice-write
        self.cache_K[self.P:self.P + self.M, :, :] = k_new.squeeze(0)
        self.cache_V[self.P:self.P + self.M, :, :] = v_new.squeeze(0)


class FlashInferBaseline(AttentionBaseline):
    def __call__(self, q, k_new, v_new):
        self.write_new_kv(k_new, v_new)
        out = flashinfer.single_prefill_with_kv_cache(
            q=q.squeeze(0),
            k=self.cache_K,
            v=self.cache_V,
            kv_layout="NHD",
            causal=CAUSAL,
        )
        return out.unsqueeze(0)


class FA2Baseline(AttentionBaseline):
    def __call__(self, q, k_new, v_new):
        self.write_new_kv(k_new, v_new)
        # FA expects (B, S, H, D), unsqueeze the cache as a view
        full_K = self.cache_K.unsqueeze(0)
        full_V = self.cache_V.unsqueeze(0)
        return flash_attn.flash_attn_func(q, full_K, full_V, causal=CAUSAL)


class FA3Baseline(AttentionBaseline):
    def __call__(self, q, k_new, v_new):
        self.write_new_kv(k_new, v_new)
        full_K = self.cache_K.unsqueeze(0)
        full_V = self.cache_V.unsqueeze(0)
        return flash_attn_interface.flash_attn_func(q, full_K, full_V, causal=CAUSAL)


class TorchSDPABaseline(AttentionBaseline):
    def __call__(self, q, k_new, v_new):
        self.write_new_kv(k_new, v_new)
        # SDPA wants (B, H, S, D)
        full_K = self.cache_K.unsqueeze(0).transpose(1, 2)
        full_V = self.cache_V.unsqueeze(0).transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), full_K, full_V, is_causal=CAUSAL,
        )
        return o.transpose(1, 2).contiguous()


if __name__ == "__main__":
    args = get_args()
    q_len, kv_len, nheads = args.m, args.n, args.k
    past_len = kv_len - q_len

    # past_len prefix already in cache
    # q_len new tokens (k_new/v_new) get written into cache[past_len:past_len + q_len]
    # attention runs against the post-write full cache

    past_K_64 = get_normal_bernoulli(
        (past_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )
    past_V_64 = get_normal_bernoulli(
        (past_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )

    k_new_64 = get_normal_bernoulli(
        (q_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )
    v_new_64 = get_normal_bernoulli(
        (q_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )

    q_64 = get_normal_bernoulli(
        (BATCH, q_len, nheads, HEAD_DIM),
        dtype=torch.float64,
    )

    # post-write full cache used by the fp64 reference
    full_K_64 = torch.cat([past_K_64, k_new_64], dim=0)
    full_V_64 = torch.cat([past_V_64, v_new_64], dim=0)

    # kernels run in bf16
    full_K = full_K_64.to(torch.bfloat16)
    full_V = full_V_64.to(torch.bfloat16)
    q = q_64.to(torch.bfloat16)

    # new K/V passed into each backend, written into the cache clone during forward
    k_new = k_new_64.to(torch.bfloat16).unsqueeze(0)
    v_new = v_new_64.to(torch.bfloat16).unsqueeze(0)

    ref = naive_attention(
        q_64,
        full_K_64.unsqueeze(0),
        full_V_64.unsqueeze(0),
    )

    # each baseline holds its own copy of the cache so writes do not interfere
    def make_cache() :
        return full_K.clone(), full_V.clone()

    pairs = []
    for name, cls in [
        ("attn_torch_sdpa", TorchSDPABaseline),
        ("attn_flashinfer", FlashInferBaseline),
        ("attn_fa2",        FA2Baseline),
        ("attn_fa3",        FA3Baseline),
    ]:
        cK, cV = make_cache()
        baseline = cls(M=q_len, P=past_len, H=nheads, D=HEAD_DIM, cache_K=cK, cache_V=cV)
        pairs.append((name, baseline))

    tensors = (q, k_new, v_new)
    outputs = []
    for name, baseline in pairs:
        out = ExperimentOutput(name, q_len, kv_len, nheads)
        out.run(baseline, tensors, ref)
        outputs.append(out)

    if args.to_csv:
        for out in outputs:
            print(ExperimentOutput.list_to_csv(out.values()))
    else:
        print(ExperimentOutput.header())
        for out in outputs:
            print(out.values())
