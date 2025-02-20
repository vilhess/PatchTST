import torch
import torch.nn as nn 

class TokenEmbedding(nn.Module):
    def __init__(self,c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
    
    def forward(self, x):
        x = self.tokenConv(x.transpose(1, 2)).transpose(1, 2)
        return x
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, cat_size, d_model):
        super().__init__()
        self.emb = nn.Linear(cat_size, d_model, bias=False)
    def forward(self, x):
        return self.emb(x)
    
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, cat_size, d_model, dp=0.1):
        super().__init__()
        self.cat_size = cat_size
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        if cat_size is not None:
            self.temporal_embedding = TimeFeatureEmbedding(cat_size=cat_size, d_model=d_model)
        self.dp = nn.Dropout(dp)
    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) 
        if self.cat_size is not None:
            x = x + self.temporal_embedding(x_mark)
        return x