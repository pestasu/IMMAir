import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class EarlyAttnTransformer(nn.Module):
    def __init__(self, args, feat_dim_a, feat_dim_m, feat_dim_o):
        super(EarlyAttnTransformer, self).__init__()
        embed_dim = feat_dim_a + feat_dim_m + feat_dim_o
        out_dim = args.num_classes if args.train_mode == "classification" else 1
        self.early_mm = Transformer(args, d_model=embed_dim, out_dim=out_dim, EMBED=True, 
                                attn_dropout=args.attn_dropout, layers=args.num_layers)
        # weight for each modality
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, inferproj_x_a, inferproj_x_m, inferproj_x_o, x_mark=None):
        '''
        inputs: [batch, seq_len, feat_dim]
        output2: [batch, pred_len]
        '''
        B, L, D = inferproj_x_a.shape
        _, T, _ = inferproj_x_o.shape
        zeros_x_a = torch.zeros(B, T, D).to(inferproj_x_a.device)
        padded_x_a = torch.cat((inferproj_x_a, zeros_x_a), dim=1)

        zeros_x_m = torch.zeros(B, T, D).to(inferproj_x_m.device)
        padded_x_m = torch.cat((inferproj_x_m, zeros_x_m), dim=1)

        zeros_x_o = torch.zeros(B, L, D).to(inferproj_x_o.device)
        padded_x_o = torch.cat((zeros_x_o, inferproj_x_o), dim=1)

        # Calculate attention weights
        combined = torch.cat((padded_x_a, padded_x_m, padded_x_o), dim=-1)

        attention_weights = self.attention(combined)

        attention_weights = attention_weights.unsqueeze(2)
        weighted_x_a = padded_x_a * attention_weights[..., 0]
        weighted_x_m = padded_x_m * attention_weights[..., 1]
        weighted_x_o = padded_x_o * attention_weights[..., 2]

        weighted_combined = torch.cat((weighted_x_a, weighted_x_m, weighted_x_o), dim=-1)

        hn_mem = self.early_mm(weighted_combined)

        return hn_mem

class LateAttnTransformer(nn.Module):
    def __init__(self, args, feat_dim_a, feat_dim_m, feat_dim_o):
        super(LateAttnTransformer, self).__init__()
        self.pred_len = args.pred_len
        combined_dims = feat_dim_a + feat_dim_m + feat_dim_o
        self.feat_dim_a = feat_dim_a
        self.feat_dim_m = feat_dim_m
        self.feat_dim_o = feat_dim_o

        self.late_a = self.get_network(args, self_type='a', num_layers=3)
        self.late_m = self.get_network(args, self_type='m', num_layers=2)
        self.late_o = self.get_network(args, self_type='o', num_layers=2)

        output_dim = args.num_classes if args.train_mode == "classification" else 1
        self.out_layer = nn.Linear(combined_dims, output_dim) 

        self.attention = nn.Sequential(
            nn.Linear(combined_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )


    def get_network(self, args, self_type='a', num_layers=0):
        if self_type in ['a']:
            embed_dim, attn_dropout, fusion_type, EMBED = self.feat_dim_a, args.attn_dropout, 'hierch', True
        elif self_type in ['m']:
            embed_dim, attn_dropout, fusion_type, EMBED = self.feat_dim_m, args.attn_dropout_m, 'hierch', True
        elif self_type in ['o']: 
            embed_dim, attn_dropout, fusion_type, EMBED = self.feat_dim_o, args.attn_dropout_o, 'late', False
        else:
            raise ValueError("Unknown network type")

        return Transformer(args, d_model=embed_dim, out_dim=embed_dim, EMBED=EMBED, fusion_type=fusion_type,
                                attn_dropout=attn_dropout, layers=max(args.num_layers, num_layers))
    
    def forward(self, inferproj_x_a, inferproj_x_m, inferproj_x_o, x_mark=None):

        B, L, D = inferproj_x_a.shape

        zeros_x_a = torch.zeros(B, self.pred_len, D).to(inferproj_x_a.device)
        padded_x_a = torch.cat((inferproj_x_a, zeros_x_a), dim=1)

        zeros_x_m = torch.zeros(B, self.pred_len, D).to(inferproj_x_m.device)
        padded_x_m = torch.cat((zeros_x_m, inferproj_x_m), dim=1)
        feat_a = self.late_a(padded_x_a)
        feat_m = self.late_m(padded_x_m)
        feat_o = self.late_o(inferproj_x_o)

        combined = torch.cat((feat_a, feat_m, feat_a), dim=-1)
        attention_weights = self.attention(combined)
        attention_weights = attention_weights.unsqueeze(2)
        weighted_x_a = feat_a * attention_weights[..., 0]
        weighted_x_m = feat_m * attention_weights[..., 1]
        weighted_x_o = feat_o * attention_weights[..., 2]

        weighted_combined = torch.cat((weighted_x_a, weighted_x_m, weighted_x_o), dim=-1)
        output = self.out_layer(weighted_combined)

        return output

class CrossAttnTransformer(nn.Module):
    def __init__(self, args, feat_dim_a, feat_dim_m, feat_dim_o):
        super(CrossAttnTransformer, self).__init__()

        self.feat_dim_a = feat_dim_a
        self.feat_dim_m = feat_dim_m
        self.feat_dim_o = feat_dim_o
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        # Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.cross_a_with_a = self.get_network(args, self_type='a')
        self.cross_a_with_m = self.get_network(args, self_type='am')
        self.cross_a_with_o = self.get_network(args, self_type='ao')
        
        self.cross_m_with_m = self.get_network(args, self_type='m')
        self.cross_m_with_a = self.get_network(args, self_type='ma')
        self.cross_m_with_o = self.get_network(args, self_type='mo')
        
        self.cross_o_with_o = self.get_network(args, self_type='o')      
        self.cross_o_with_a = self.get_network(args, self_type='oa')
        self.cross_o_with_m = self.get_network(args, self_type='om')
        
        self.cross_a_mem = self.get_network(args, self_type='a_mem', num_layers=3)
        self.cross_o_mem = self.get_network(args, self_type='o_mem', num_layers=3)
        self.cross_m_mem = self.get_network(args, self_type='m_mem', num_layers=3)

        combined_dims = self.feat_dim_a*3 + self.feat_dim_m*3 + self.feat_dim_o*3 

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # Projection layers
        self.out_layer = nn.Linear(combined_dims, output_dim)        
        
    def get_network(self, args, self_type='a', num_layers=0):
        if self_type in ['a', 'm', 'o', 'am', 'ao', 'ma', 'mo', 'oa', 'om']:
            d_model, out_dim, fusion_type, EMBED = args.dmodel_nheads[0], self.seq_len, 'cross', True
        elif self_type in ['a_mem', 'm_mem', 'o_mem']:
            d_model, out_dim, fusion_type, EMBED = args.dmodel_nheads[0], self.pred_len, 'cross', True
        else:
            raise ValueError("Unknown network type")

        return Transformer(args, d_model=d_model, out_dim=out_dim, EMBED=EMBED, 
                                fusion_type=fusion_type, layers=max(args.num_layers, num_layers))

    def forward(self, inferproj_x_a, inferproj_x_m, inferproj_x_o, mark=None):
        # (meo,outphoto) --> aqi
        hn_a_with_a = self.cross_a_with_a(inferproj_x_a, inferproj_x_a, inferproj_x_a)
        hn_a_with_m = self.cross_a_with_m(inferproj_x_a, inferproj_x_m, inferproj_x_m)
        hn_a_with_o = self.cross_a_with_o(inferproj_x_a, inferproj_x_o, inferproj_x_o)
        hn_a_mem = torch.cat([hn_a_with_a, hn_a_with_m, hn_a_with_o], dim=-1) 
        hn_a = self.cross_a_mem(hn_a_mem, hn_a_mem, hn_a_mem)
        
        # (aqi,outphoto) --> meo
        hn_m_with_m = self.cross_m_with_m(inferproj_x_m, inferproj_x_m, inferproj_x_m)
        hn_m_with_a = self.cross_m_with_a(inferproj_x_m, inferproj_x_a, inferproj_x_a)
        hn_m_with_o = self.cross_m_with_o(inferproj_x_m, inferproj_x_o, inferproj_x_o)
        hn_m_mem = torch.cat([hn_m_with_m, hn_m_with_a, hn_m_with_o], dim=-1) 
        hn_m = self.cross_m_mem(hn_m_mem, hn_m_mem, hn_m_mem)
        
        # (aqi,meo) --> outphoto
        hn_o_with_o = self.cross_o_with_o(inferproj_x_o, inferproj_x_o, inferproj_x_o)
        hn_o_with_a = self.cross_o_with_a(inferproj_x_o, inferproj_x_a, inferproj_x_a)
        hn_o_with_m = self.cross_o_with_m(inferproj_x_o, inferproj_x_m, inferproj_x_m)
        hn_o_mem = torch.cat([hn_o_with_o, hn_o_with_a, hn_o_with_m], dim=-1) 
        hn_o = self.cross_o_mem(hn_o_mem, hn_o_mem, hn_o_mem)

        last_hs = torch.cat([hn_a, hn_m, hn_o], dim=-1)

        output = self.out_layer(last_hs)

        weight_size = self.out_layer.weight.size(1)//3
        out_a = (torch.mm(hn_a.squeeze(1), torch.transpose(self.out_layer.weight[:, :weight_size], 0, 1))
                     + self.out_layer.bias)
        out_m = (torch.mm(hn_m.squeeze(1), torch.transpose(self.out_layer.weight[:, weight_size:weight_size*2], 0, 1))
                     + self.out_layer.bias)
        out_o = (torch.mm(hn_o.squeeze(1), torch.transpose(self.out_layer.weight[:, -weight_size:], 0, 1))
                     + self.out_layer.bias)

        return out_a, out_m, out_o, output

class Transformer(nn.Module):

    def __init__(self, config, d_model, out_dim=1, attn_dropout=0.0, layers=0, 
                EMBED=True, attn_mask=False, fusion_type='', activation='gelu'):
        super(Transformer, self).__init__()
        d_ff = d_model
        factor = config.factor
        num_heads = config.dmodel_nheads[1]
        res_dropout = config.res_dropout
        self.out_dim = out_dim
        self.EMBED = EMBED
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.dropout = config.embed_dropout
        self.train_mode = config.train_mode
        self.fusion_type = fusion_type if fusion_type != '' else config.fusion_type

        # Embedding
        self.embed_inverted = DataEmbedding_inverted(self.seq_len, d_model)
        self.embed_inverted_o = DataEmbedding_inverted(self.pred_len, d_model)
        self.embed_position = PositionalEmbedding(d_model)

        # Encoder
        self.invert_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=attn_dropout), 
                        d_model, num_heads),
                    d_model,
                    d_ff,
                    res_dropout,
                    activation=activation
                ) for l in range(layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Encoder
        self.feat_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=attn_dropout),
                        d_model, num_heads),
                    d_model,
                    d_ff,
                    res_dropout,
                    activation=activation
                ) for l in range(layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(layers - 1)
            ] if 'regression' in self.train_mode else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.register_buffer('version', torch.Tensor([2]))

        # Decoder
        d_layers = 2
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=attn_dropout),
                        d_model, num_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=attn_dropout),
                        d_model, num_heads),
                    d_model,
                    d_ff,
                    res_dropout,
                    activation=activation
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, self.out_dim, bias=True)
        )
        self.register_buffer('version', torch.Tensor([2]))
        self.projection = nn.Linear(d_model, self.out_dim) 
    
    def feat_forward(self, x):
        # feat attention
        if self.EMBED:
            x = x + self.embed_position(x)
        # px, px_k, px_v = self.embedding_position(x, x_k, x_v)
        x_enc = x[:, :self.seq_len]
        x_dec = x[:, -(self.seq_len//2):]
        enc_out, _ = self.feat_encoder(x_enc)

        dec_out = self.decoder(x_dec, enc_out)

        out = dec_out[:, -self.pred_len:]

        return out

    def embedding_inverted(self, x, x_k, x_v):
        if self.embed_inverted is not None:
            x = x.permute(0, 2, 1)
            if x.shape[-1] != self.seq_len:
                embed_x = self.embed_inverted_o(x)   # Add inverted embedding
            else:
                embed_x = self.embed_inverted(x)
            embed_x = F.dropout(embed_x, p=self.dropout, training=self.training)
            if x_k is not None and x_v is not None:
                x_k = x_k.permute(0, 2, 1)
                x_v = x_v.permute(0, 2, 1)
                if x_k.shape[-1] != self.seq_len:
                    embed_x_k = self.embed_inverted_o(x_k) 
                    embed_x_v = self.embed_inverted_o(x_v)
                else:
                    embed_x_k = self.embed_inverted(x_k)  # Add inverted embedding
                    embed_x_v = self.embed_inverted(x_v)  # Add inverted embedding

                embed_x_k = F.dropout(embed_x_k, p=self.dropout, training=self.training)
                embed_x_v = F.dropout(embed_x_v, p=self.dropout, training=self.training)
            else:
                embed_x_k, embed_x_v = embed_x, embed_x
        return embed_x, embed_x_k, embed_x_v

    def time_forward(self, x, x_k, x_v):
        # feat attention
        if self.EMBED:
            x, x_k, x_v = self.embedding_inverted(x, x_k, x_v)
        
        enc_out, _ = self.invert_encoder(x, x_k, x_v)
        
        out = self.projection(enc_out).permute(0, 2, 1)

        return out

    def forward(self, x, x_k=None, x_v=None):
        # encoder and decoder layers
        if self.fusion_type == 'cross' or self.fusion_type == 'late':
            return self.time_forward(x, x_k, x_v) 
        else:
            return self.feat_forward(x) 

