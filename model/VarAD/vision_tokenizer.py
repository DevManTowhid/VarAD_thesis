from torch import nn
import torch.nn.functional as F
from .tokenizer_backbones import *
import torch


# DIMENSION_DICT = {
#     'resnet18': 448,
#     'resnet34': 448,
#     'resnet50': 1792,
#     'resnet101': 1792,
#     'resnet152': 1792,
#     'wide_resnet50_2': 1792,
#     'wide_resnet101_2': 1792,
#     'hrnet18':126,
#     'hrnet32':224,
#     'hrnet48':336
# }

DIMENSION_DICT = {
    'resnet18': [64, 128, 256],
    'resnet34': [64, 128, 256],
    'resnet50': [256, 512, 1024],
    'resnet101': [256, 512, 1024],
    'resnet152': [256, 512, 1024],
    'wide_resnet50_2': [256, 512, 1024],
    'wide_resnet101_2': [256, 512, 1024],
    'hrnet18':[18, 36, 72],
    'hrnet32':[32, 64, 128],
    'hrnet48':[48, 96, 192],
    'SAM': [256, 256, 256],
    'DINO': [384, 384, 384],
    'dinov2_vits14': [384, 384, 384],
    'dinov2_vitb14': [768, 768, 768],
    'dinov2_vitl14': [1024, 1024, 1024],
    'CLIP':  [768, 768, 768],
    'VIT':  [768, 768, 768],
    'MAE': [768, 768, 768],
}

class VisionTokenizer(nn.Module):
    def __init__(self, backbone: str,
                 hierarchies: list=[1,2,3],
                 is_proj: bool=True,
                 norm: bool=False,
                 image_size: int=256):
        super(VisionTokenizer, self).__init__()

        try:
            if backbone.count('resnet') > 0 or backbone.count('hrnet') > 0:
                self.frozen_tokenizer = eval(f'{backbone}(pretrained=True)')
            elif backbone == 'SAM':
                self.frozen_tokenizer = eval(f'{backbone}()')
            elif backbone.count('DINO') > 0:
                self.frozen_tokenizer = eval(f'{backbone}()')
            elif backbone.count('dinov2') > 0:
                self.frozen_tokenizer = DINO(backbone)
            elif backbone == 'CLIP':
                self.frozen_tokenizer = CLIP(image_size=image_size)
            elif backbone == 'VIT' or backbone == 'MAE':
                self.frozen_tokenizer = ViT_backbone(backbone=backbone, image_size=image_size)
            else:
                raise NotImplementedError
        except:
            print(f'unsupported backbone for frozen_tokenizer: {backbone}. please select from {DIMENSION_DICT.keys()}')
            raise NotImplementedError

        # self.frozen_tokenizer = DINO(backbone)

        # change the frozen_tokenizer into testing model
        for param in self.frozen_tokenizer.parameters():
            param.requires_grad = False

        self.frozen_tokenizer.eval()

        self.norm = norm
        self.hierarchies = hierarchies
        self.target_dimension = [DIMENSION_DICT[backbone][i-1] for i in hierarchies]
        self.is_proj = is_proj

        self.proj = nn.ModuleList() # feature adaptor. can significantly decrease the training difficulty

        for d in self.target_dimension:
            proj_temp = []
            proj_temp.append(nn.Conv2d(d, d,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=0))
            # proj_temp.append(nn.ReLU())
            self.proj.append(nn.Sequential(*proj_temp))

    def forward(self, x):
        # x: bs x 3 x H x W
        x = self.frozen_tokenizer(x)

        x = [x[i-1] for i in self.hierarchies]

        # TODO: introduce random in the inference stage. boosting several times.
        for indx in range(len(x)):
            x[indx] = x[indx].detach()
            if self.is_proj:
                x[indx] = self.proj[indx](x[indx]) + x[indx] # adopted
            else:
                x[indx] = x[indx]

            if self.norm:
                x[indx] = F.normalize(x[indx], dim=1)
        return x
