"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
"""
import torch
import torch.nn as nn
from .segment_anything import SamPredictor, sam_model_registry
import torch.nn.functional as F
# from sam_aggregator_neck import SAMAggregatorNeck

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LinearHead(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_class, kernel_size=3, stride=1, padding=1),
        )

        self.neck = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            LayerNorm(256, data_format="channels_first"),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            LayerNorm(256, data_format="channels_first")
        )


    def forward(self, x):
        if x.shape[1] == 1024:
            x = self.neck(x)
        x = self.conv1(x)

        # sigmoid
        x = 1 / (1 + torch.exp(-x))
        return x


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.encoder = sam_model_registry['vit_l'](checkpoint='./weights/sam_vit_l_0b3195.pth').image_encoder
        # self.encoder = sam_model_registry['vit_b'](checkpoint='SAM/checkpoint/sam_vit_b_01ec64.pth').image_encoder

        self.pos_embed = nn.Parameter(F.interpolate(
            self.encoder.pos_embed.permute(0, 3, 1, 2), 
            size=[16, 16],
            mode='bicubic').permute(0, 2, 3, 1))

        self.patch_embed = torch.nn.Sequential(
            self.encoder.patch_embed,
        )
        self.blocks = self.encoder.blocks
        self.neck = self.encoder.neck
        self.feat_list = [8, 16 ,24]


    def forward(self, x):
        patch_tokens = []

        x = self.patch_embed(x)
        x = x + self.pos_embed

        for idx, blk in enumerate(self.blocks):
            x = blk(x)

            if (idx+1) in self.feat_list:

                temp = self.neck(x.permute(0, 3, 1, 2))
                patch_tokens.append(temp)

        return patch_tokens
    
