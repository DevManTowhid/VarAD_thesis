from torch import nn
import torch.nn.functional as F
from .tokenizer_backbones import *
import torch
from .lora import LoRA
import math

DIMENSION_DICT = {
    'SAM': [256, 256, 256],
    'DINO': [384, 384, 384],
    'dinov2_vits14': [384, 384, 384],
    'dinov2_vitb14': [768, 768, 768],
    'dinov2_vitl14': [1024, 1024, 1024],
}

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    



class VisionTokenizer(nn.Module):
    def __init__(self, backbone: str,
                 hierarchies: list=[1,2,3],
                 adapter: list=['linear'],
                 norm: bool=False,
                 r: int=3):
        super(VisionTokenizer, self).__init__()

        self.norm = norm
        self.hierarchies = hierarchies
        self.target_dimension = [DIMENSION_DICT[backbone][i-1] for i in hierarchies]

        try:
            if backbone.count('DINO') > 0:
                self.frozen_tokenizer = eval(f'{backbone}()')
            elif backbone.count('dinov2') > 0:
                self.frozen_tokenizer = DINO(backbone)
            else:
                raise NotImplementedError
        except:
            print(f'unsupported backbone for frozen_tokenizer: {backbone}. please select from {DIMENSION_DICT.keys()}')
            raise NotImplementedError

        # change the frozen_tokenizer into testing model
        for param in self.frozen_tokenizer.parameters():
            param.requires_grad = False

        self.frozen_tokenizer.eval()

        self.is_proj = False

        self.proj = nn.ModuleList()  # feature adaptor. can significantly decrease the training difficulty

        # default to use linear
        for d in self.target_dimension:
            proj_temp = []
            proj_temp.append(nn.Conv2d(d, d,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=0))
            # proj_temp.append(nn.ReLU())
            self.proj.append(nn.Sequential(*proj_temp))

        if 'linear' in adapter:
            self.is_proj = True


        if 'resblock' in adapter:
            self.is_proj = True
            self.proj = nn.ModuleList()  # feature adaptor. can significantly decrease the training difficulty

            for d in self.target_dimension:
                proj_temp = []
                proj_temp.append(ResBlock(d, d))
                self.proj.append(nn.Sequential(*proj_temp))

        if 'lora' in adapter:
            self.lora_layers = list(range(len(self.frozen_tokenizer.dino.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.frozen_tokenizer.dino.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()


    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
            w_a.requires_grad = True

        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)
            w_b.requires_grad = True

    def forward(self, x):
        # x: bs x 3 x H x W
        x = self.frozen_tokenizer(x)

        x = [x[i-1] for i in self.hierarchies]

        for indx in range(len(x)):
            x[indx] = x[indx].detach()
            if self.is_proj:
                x[indx] = self.proj[indx](x[indx]) + x[indx] # adopted
            else:
                x[indx] = x[indx]

            if self.norm:
                x[indx] = F.normalize(x[indx], dim=1)
        return x
