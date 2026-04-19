import torch.nn as nn
import torch

class Vanilla(nn.Module):
    """
    Taken directly from Trinity
    """
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        """
        Docstring for __init__
        
        :param M: num. of new tokens(e.g. 16)
        :param N: model dim(e.g. 4096)
        :param D: embedding dim(e.g. 128, heads=N/D)
        :param P: num. of existing tokens(e.g. 1008)
        :param cache_K: tensor
        :param cache_V: tensor
        :param W_q: tensor
        :param W_k: tensor
        :param W_v: tensor
        :param device: -
        :param dtype: -
        """
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):

        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # q = q.transpose(0, 1)

        # scores = torch.matmul(q, self.cache_K.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)
        
        # output = torch.matmul(weights, self.cache_V)
        # output = output.view(self.M, self.H * self.D)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        # weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        # output = output.reshape(self.M, self.H * self.D)
        return output