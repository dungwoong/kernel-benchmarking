import torch
from fla_la import fla_linear_attention
from torch_la import simple_gla

# This is as close as the two attention implementations get...

torch.manual_seed(3)

def get_rmse(ref: torch.Tensor, o: torch.Tensor):
    assert o.dtype == ref.dtype
    mse = torch.nn.functional.mse_loss(o, ref, reduction='mean')
    rmse = mse.sqrt().item()
    return rmse

def rmsnorm(a: torch.Tensor, eps=1e-5):
    return torch.nn.functional.rms_norm(a, normalized_shape=(a.shape[-1],), eps=eps)


if __name__ == '__main__':
    B, H, T, K, V = 2, 4, 128, 32, 64
    device = "cuda"
    dtype = torch.float32
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    gamma = torch.rand(H, device=device, dtype=dtype) * -0.01
    scale = K ** -0.5
    o, _ = fla_linear_attention(q, k, v, g_gamma=gamma, scale=scale)
    o2 = simple_gla(q, k, v, gamma=gamma, scale=scale)
    mean_output = torch.mean(o.abs()).item()
    max_output = torch.max(o2.abs()).item()
    max_err = torch.max((o - o2).abs()).item()
    rmse = get_rmse(o, o2)
    print(f'{mean_output=} {max_output=}, {max_err=}, {rmse=}')
    assert torch.allclose(o, o2, atol=1e-1, rtol=1e-2)
