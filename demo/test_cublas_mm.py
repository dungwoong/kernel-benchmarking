import torch
from cublas_mm import matmul
from triton.testing import do_bench


a = torch.randn((8192, 8192), dtype=torch.bfloat16, device='cuda')
b = torch.randn((8192, 8192), dtype=torch.bfloat16, device='cuda')
c = matmul(a, b)
ref = a @ b.t()

print(c - ref)

ms_1 = do_bench(lambda: matmul(a, b))
ms_2 = do_bench(lambda: a @ b.t())
print(f'{ms_1}, {ms_2}')