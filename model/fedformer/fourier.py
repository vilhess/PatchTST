import torch 
import torch.nn as nn 
import numpy as np 

def get_frequency_modes(seq_len, modes):
    modes = min(modes, seq_len//2)
    index = list(range(0, seq_len//2))
    np.random.shuffle(index)
    index = index[:modes]
    index.sort()
    return index

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, n_heads, modes):
        super().__init__()
        self.index = get_frequency_modes(seq_len, modes)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, in_channels // n_heads, out_channels // n_heads, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)
    
    def forward(self, q, k, v):
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    
class FourierCrossAttention(nn.Module):
    def __init__(self, in_c, out_c, seq_len_q, seq_len_kv, n_heads, modes):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.index_q = get_frequency_modes(seq_len_q, modes=modes)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes)
        self.scale = 1/ (in_c*out_c)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(n_heads, in_c // n_heads, out_c // n_heads, len(self.index_q), dtype=torch.cfloat))
        
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)
    
    def forward(self, q, k, v):
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xk.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xv_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xv.device, dtype=torch.cfloat)
        xv_ft = torch.fft.rfft(xv, dim=-1)
        for i, j in enumerate(self.index_kv):
            xv_ft_[:, :, :, i] = xv_ft[:, :, :, j]

        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        xqk_ft = xqk_ft.tanh()
        xqkv_ft = torch.einsum('bhxy,bhey->bhex', xqk_ft, xv_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L//2 +1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:,:,:,j] = xqkvw[:,:,:,i]
        out = torch.fft.irfft(out_ft /self.in_c / self.out_c, n=xq.size(-1))
        return out