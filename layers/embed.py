import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import RoIPool
from torch.nn.utils import weight_norm
from torchvision.ops.deform_conv import DeformConv2d
from ipaddress import ip_address

class AirEmbedding(nn.Module):
    def __init__(self, args):
        super(AirEmbedding, self).__init__()
        self.fusion_type = args.fusion_type
        self.embed_dim = args.dst_feat_dim
        self.seq_len = args.seq_len
        self.photo_dropout = args.photo_dropout
        self.origfeat_dim_a, self.origfeat_dim_m, _ = args.feat_dims
        self.pretrained_type = args.pretrained_type

        self.proj_a = nn.LSTM(self.origfeat_dim_a, self.embed_dim, 1, batch_first=True)
        self.proj_m = nn.LSTM(self.origfeat_dim_m, self.embed_dim, 1, batch_first=True)
        
        if not args.use_finetune:
            if self.pretrained_type == 'resnet':
                pre_model = models.resnet50(pretrained=True)
                children = nn.Sequential(*(list(pre_model.children())[:-2]))
                self.pre_extractor = nn.Sequential(
                    children,
                    nn.Conv2d(2048, 1024, kernel_size=1),
                    nn.ReLU(),  
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.origfeat_dim_o = 1024
            elif self.pretrained_type == 'custom':
                self.pre_extractor = CustomHazeExtractor(args.train_mode)
                self.pre_extractor.set_return_feat(True)
                pretrained_weights = torch.load(f'pretrained/{args.pretrained_type}-{args.train_mode}-pretrained.pth')
                state_dict = self.pre_extractor.state_dict()
                new_state_dict = {}
                for k, v in pretrained_weights.items():
                    k = k.replace('module.', '')
                    new_state_dict[k] = v
                state_dict.update(new_state_dict)
                self.pre_extractor.load_state_dict(state_dict)
                self.origfeat_dim_o = 64


        self.proj_o = nn.Conv2d(self.origfeat_dim_o, self.embed_dim, kernel_size=3, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.pred_embed_size, self.embed_dim, kernel_size=1, padding=0, bias=False)
        self.proj_m = nn.Conv1d(self.pred_embed_size, self.embed_dim, kernel_size=1, padding=0, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, aqi, meo, photo):
        # project three data to feature space
        h_o = F.dropout(self.pre_extractor(photo), p=self.photo_dropout, training=self.training) # [batch_size, cnn_feature_dim, H', W']
        h_a, _ = self.proj_a(aqi)
        h_m, _ = self.proj_m(meo)

        feat_o = self.avgpool(self.proj_o(h_o)).flatten(2)
        feat_a, feat_m = h_a.transpose(1, 2), h_m.transpose(1, 2)

        return feat_a, feat_m, feat_o

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
        )

