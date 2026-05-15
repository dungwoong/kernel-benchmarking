import torch

from profile_utils import get_normal_bernoulli, get_attention_args
from attention_decode import (
    BATCH,
    HEAD_DIM,
    FlashInferBaseline,
    FA2Baseline,
    FA2FlashDecodeBaseline,
    FA3Baseline,
    FA3FlashDecodeBaseline,
    TorchSDPABaseline,
)

torch.manual_seed(18)


if __name__ == "__main__":
    args = get_attention_args()
    q_len, kv_len, nheads = args.q_len, args.kv_len, args.nheads
    past_len = kv_len - q_len

    past_K = get_normal_bernoulli(
        (past_len, nheads, HEAD_DIM), ncu=True
    )
    past_V = get_normal_bernoulli(
        (past_len, nheads, HEAD_DIM), ncu=True
    )

    k_new_3d = get_normal_bernoulli((q_len, nheads, HEAD_DIM), ncu=True)
    v_new_3d = get_normal_bernoulli((q_len, nheads, HEAD_DIM), ncu=True)

    full_K = torch.cat([past_K, k_new_3d], dim=0)
    full_V = torch.cat([past_V, v_new_3d], dim=0)

    q = get_normal_bernoulli(
        (BATCH, q_len, nheads, HEAD_DIM), ncu=True
    )
    k_new = k_new_3d.unsqueeze(0)
    v_new = v_new_3d.unsqueeze(0)

    for cls in [
        TorchSDPABaseline,
        FlashInferBaseline,
        FA2Baseline,
        FA2FlashDecodeBaseline,
        FA3Baseline,
        FA3FlashDecodeBaseline,
    ]:
        cK = full_K.clone()
        cV = full_V.clone()
        baseline = cls(
            M=q_len, P=past_len, H=nheads, D=HEAD_DIM,
            cache_K=cK, cache_V=cV,
        )
        baseline(q, k_new, v_new)
