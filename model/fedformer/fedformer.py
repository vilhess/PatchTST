# Modified Code of the original repository focus on the Fourier version; 
# corrected some things:
# - works with different head size
# - correction fourier cross attention
# - works for data without categorical features known

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from model.fedformer.layers import Encoder, EncoderLayer, Decoder, DecoderLayer, AutoCorrelationLayer
from model.fedformer.series import series_decomp, series_decomp_multi, myLayerNorm
from model.fedformer.embed import DataEmbedding_wo_pos
from model.fedformer.fourier import FourierBlock, FourierCrossAttention


class FEDformer(nn.Module):
    def __init__(self, seq_len=96, label_len=48, pred_len=96, modes=32, enc_in=7, dec_in=7,
                c_out=7, cat_size=4, moving_avg=[4, 8], e_layers=4, d_layers=1, n_heads=6,
                d_model=256, d_ff=512, dp=0.05):
        
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.cat_size = cat_size

        if isinstance(moving_avg, list):
            self.decomp = series_decomp_multi(moving_avg)
        else:
            self.decomp = series_decomp(moving_avg)

        self.enc_embedding = DataEmbedding_wo_pos(cat_size=cat_size, c_in=enc_in, d_model=d_model, dp=dp)
        self.dec_embedding = DataEmbedding_wo_pos(cat_size=cat_size, c_in=dec_in, d_model=d_model, dp=dp)

        encoder_self_attn = FourierBlock(in_channels=d_model, out_channels=d_model, seq_len=seq_len, n_heads=n_heads, modes=modes)
        decoder_self_attn = FourierBlock(in_channels=d_model, out_channels=d_model, seq_len=label_len + pred_len, n_heads=n_heads, modes=modes)
        decoder_cross_attn = FourierCrossAttention(in_c=d_model, out_c=d_model, seq_len_q=label_len + pred_len, seq_len_kv=seq_len, n_heads=n_heads, modes=modes)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_attn, d_model=d_model, n_heads=n_heads),
                    d_model, d_ff, moving_avg, dp
                ) for l in range(e_layers)
            ],
            norm_layers=myLayerNorm(d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_attn, d_model=d_model, n_heads=n_heads),
                    AutoCorrelationLayer(decoder_cross_attn, d_model=d_model, n_heads=n_heads),
                    d_model, c_out, d_ff, moving_avg, dp
                ) for l in range(d_layers)
            ],
            norm_layer=myLayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )
    
    def forward(self, x_enc, x_enc_mark=None, x_dec_mark=None):
        # x_enc: B,seq_len,F ; x_enc_mark: B,seq_len,F_mark ; x_dec_mark: B,label+pred len,F_mark ; 

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1) # B,1,F -> B, pred_len, F
        seasonal_init, trend_init = self.decomp(x_enc) # B, seq_len, F
        trend_init = torch.cat([trend_init[:,-self.label_len:,:], mean], dim=1) # B, label+pred len, F
        seasonal_init = F.pad(seasonal_init[:,-self.label_len:, :], (0, 0, 0, self.pred_len)) # # B, label+pred len, F (rajoute des 0 de label_len jusqua pred_len)
        enc_in = self.enc_embedding(x_enc, x_enc_mark) # B, seq_len, D
        enc_out = self.encoder(enc_in) # B, seq_len, D
        dec_in = self.dec_embedding(seasonal_init, x_dec_mark) # B, label+pred len, D
        seasonal_part, trend_part = self.decoder(dec_in, enc_out, trend=trend_init)
        decoder_out = trend_part + seasonal_part
        return decoder_out[:,-self.pred_len:, :]
    

if __name__=="__main__":

    seq_len=64
    label_len=32
    pred_len=64 
    enc_in=7
    dec_in=7
    cat_size=4
    c_out=7
    modes=32
    e_layers=2
    d_layers=2 
    moving_avg=[16, 32]
    d_model=256
    n_heads=4
    d_ff=512
    dp=0.05

    model = FEDformer(modes=modes, seq_len=seq_len, label_len=label_len, pred_len=pred_len, e_layers=e_layers, moving_avg=moving_avg,
                      enc_in=enc_in, d_model=d_model, d_ff=d_ff, dp=dp, n_heads=n_heads, c_out=c_out, d_layers=d_layers, 
                      dec_in=dec_in, cat_size=cat_size)
    
    x_enc = torch.randn(128, seq_len, enc_in)
    x_enc_mark = torch.randn(128, seq_len, cat_size)
    x_dec_mark = torch.randn(128, label_len+pred_len, cat_size)

    out = model(x_enc, x_enc_mark, x_dec_mark)
    print(out.shape)