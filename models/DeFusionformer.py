import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Series_Decomposition import series_decomp
from layers.Embed import DataEmbedding


class DeFusionformer(nn.Module):
    """
    DeFusionformer
    """
    def __init__(self, configs):
        super(DeFusionformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        #Decomposition
        # Decomp
        kernel_size_L = configs.moving_avg_L
        kernel_size_S = configs.moving_avg_S

        self.decomp_L = series_decomp(kernel_size_L)
        self.decomp_S = series_decomp(kernel_size_S)

        self.Weight_L = nn.Linear(1, configs.d_model)
        self.Weight_S = nn.Linear(1, configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_L_enc, x_L_mark_enc, x_S_enc, x_S_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # decomp init_mid
        seasonal_L_init, trend_L_init = self.decomp_L(x_L_enc)

        # decomp init_short
        seasonal_S_init, trend_S_init = self.decomp_S(x_S_enc)

        weight_L = self.Weight_L(trend_L_init)
        weight_S = self.Weight_S(trend_S_init)

        enc_L_out = self.enc_embedding(seasonal_L_init, x_L_mark_enc)
        enc_L_out, attns_L = self.encoder(enc_L_out, attn_mask=enc_self_mask)

        enc_S_out = self.enc_embedding(seasonal_S_init, x_S_mark_enc)
        enc_S_out, _ = self.encoder(enc_S_out, attn_mask=enc_self_mask)

        enc_long = weight_L + enc_L_out
        enc_short = weight_S + enc_S_out

        enc = torch.concat([enc_long, enc_short], dim=1)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], _
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

