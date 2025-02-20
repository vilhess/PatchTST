import torch 
import torch.nn as nn 
from model.fedformer.series import series_decomp, series_decomp_multi

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_correlation(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)
        return self.out_projection(out)
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, moving_avg, dp):
        super().__init__()
        d_ff = d_ff 
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(kernel_sizes=moving_avg)
            self.decomp2 = series_decomp_multi(kernel_sizes=moving_avg)
        else:
            self.decomp1 = series_decomp(kernel_size=moving_avg)
            self.decomp2 = series_decomp(kernel_size=moving_avg)

        self.dp = nn.Dropout(dp)
        self.activation = nn.GELU()

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dp(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dp(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dp(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(y+x)
        return res
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layers):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layers

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)   
        x = self.norm(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, c_out, d_ff, moving_avg, dp):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(kernel_sizes=moving_avg)
            self.decomp2 = series_decomp_multi(kernel_sizes=moving_avg)
            self.decomp3 = series_decomp_multi(kernel_sizes=moving_avg)
        else:
            self.decomp1 = series_decomp(kernel_size=moving_avg)
            self.decomp2 = series_decomp(kernel_size=moving_avg)
            self.decomp3 = series_decomp(kernel_size=moving_avg)
        
        self.dp = nn.Dropout(dp)
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1, padding_mode="circular", bias=False)
        self.activation = nn.GELU()

    def forward(self, x, cross):
        x = x + self.dp(self.self_attn(x, x, x))
        x, trend1 = self.decomp1(x)
        x = x + self.dp(self.cross_attn(x, cross, cross))
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dp(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dp(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x+y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.proj(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
    
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer, projection):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.proj = projection

    def forward(self, x, cross, trend):
        for layer in self.layers:
            x, residual_trend = layer(x, cross)
            trend = trend + residual_trend
        x = self.norm(x)
        x = self.proj(x)
        return x, trend