import os
import torch
import helion
from helion import Config
import helion.language as hl


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "helion"
)

_kernel_cache = {}


def config_key(m, n, k, dtype, effort="quick"):
    # Naming scheme: quick by default, full if full effort
    base = f"matmul_nt_{m}_{n}_{k}_{str(dtype).split('.')[-1]}"
    return base if effort == "quick" else f"{base}_{effort}"


def cache_path(m, n, k, dtype, effort="quick"):
    return os.path.join(CACHE_DIR, f"{config_key(m, n, k, dtype, effort)}.json")


def _matmul_nt_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    n, _ = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_n, tile_k].T)
        out[tile_m, tile_n] = acc.to(a.dtype)
    return out


def autotune(a: torch.Tensor, b: torch.Tensor, effort: str = "quick") -> str:
    """Autotune at the specified effort and save the Config to .cache/helion/
    """
    m, k = a.shape
    n = b.shape[0]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(m, n, k, a.dtype, effort)
    tuner = helion.kernel(static_shapes=True, autotune_effort=effort)(_matmul_nt_fn)
    best = tuner.autotune((a, b))
    best.save(path)
    return path


def _resolve_path(m, n, k, dtype, effort):
    # quick/full effort -> use that exact file
    # None -> prefer full if it exists, else quick
    if effort in ("quick", "full"):
        return cache_path(m, n, k, dtype, effort)
    full = cache_path(m, n, k, dtype, "full")
    return full if os.path.exists(full) else cache_path(m, n, k, dtype, "quick")


def _get_kernel(a: torch.Tensor, b: torch.Tensor, effort=None):
    m, k = a.shape
    n = b.shape[0]
    path = _resolve_path(m, n, k, a.dtype, effort)
    if path not in _kernel_cache:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No Helion config at {path}. "
            )
        _kernel_cache[path] = helion.kernel(config=Config.load(path), static_shapes=True)(_matmul_nt_fn)
    return _kernel_cache[path]


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes a @ b.T for a=(m, k), b=(n, k). Loads a pre-tuned Config, does not autotune

    Which config: HELION_GEMM_EFFORT=quick|full forces that effort
    if none specified then defaults to full, fall back on quick
    """
    assert a.ndim == 2 and b.ndim == 2, "Expected two matrices"
    assert a.is_cuda and b.is_cuda, "Expected CUDA tensors"
    assert a.shape[1] == b.shape[1], "Incompatible reduction dimensions"
    assert a.dtype == b.dtype, "Input dtypes must match"
    effort = os.environ.get("HELION_GEMM_EFFORT")  # 'quick', 'full', or None (auto)
    return _get_kernel(a, b, effort)(a, b)
