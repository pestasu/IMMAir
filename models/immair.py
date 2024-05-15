import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from .condiff_net import cdr_layer

    
class iMMAir(nn.Module):
    def __init__(self, args):
        super(iMMAir, self).__init__()
        self.use_embed = args.use_embed
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.origfeat_dim_a, self.origfeat_dim_m, self.origfeat_dim_o = args.feat_dims
        self.feat_dim_a = self.feat_dim_m = self.feat_dim_o = args.dst_feat_dim
        self.output_dropout = args.output_dropout
        self.photo_dropout = args.photo_dropout
        self.train_mode = args.train_mode

        # 1. Get unimodel encoding layer
        self.airembed = AirEmbedding(args)

        # 2. Get conditional diffuison-based recovery layer
        self.cdr_layer = cdr_layer(args, self.feat_dim_a, self.feat_dim_m, self.feat_dim_o)

        # 3. Get fusion model layer
        self.air_mmt = self.get_model_fusion(args)

    def get_model_fusion(self, args):
        # 2. Cross-Attentions
        if args.fusion_type == 'cross':
            return CrossAttnTransformer(args, self.feat_dim_a, self.feat_dim_m, self.feat_dim_o)
        elif args.fusion_type == 'early':
            return EarlyAttnTransformer(args, self.feat_dim_a, self.feat_dim_m, self.feat_dim_o)
        elif args.fusion_type == 'late':
            return LateAttnTransformer(args, self.feat_dim_a, self.feat_dim_m, self.feat_dim_o)
        else:
            raise ValueError("Unknown Multimodel fusion type!")

    def forward(self, aqi, meo, photo=None, label=None, mark=None, num_modal=None, is_train=True):
        '''
            aqi: [batch_size, sequence_len, feature_dims]
            meo: [batch_size, sequence_len, feature_dims]
            photo: [batch_size, patch_len, feature_dims] or [batch_size, pred_len, in_channel, feature_dims, feature_dims]
        '''
        x_a, x_m, x_o = aqi, meo[:, :self.seq_len], photo.squeeze(1)

        with torch.no_grad():
            z_a, z_m, z_o = self.airembed(x_a, x_m, x_o)


        proj_x_a, proj_x_m, proj_x_o = z_a, z_m, z_o

        inferproj_x_a, inferproj_x_m, inferproj_x_o, loss_trans, loss_rec, ava_modal_idx \
            = self.cdr_layer(proj_x_a, proj_x_m, proj_x_o, label, num_modal, is_train)

        out_a, out_m, out_o, output = self.air_mmt(inferproj_x_a, inferproj_x_m, inferproj_x_o, mark)
        
        res = {
            'ava_modal_idx': ava_modal_idx,
            'loss_trans':loss_trans,
            'loss_rec': loss_rec,
            'uni_hat': [out_a, out_m, out_o],
            'y_hat': output
        }
        return res
 