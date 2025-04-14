import numpy as np

from .custom_clip import create_model_and_transforms
from torch import nn

class CLIP_backbone(nn.Module):

    def __init__(self, image_size):
        super(CLIP_backbone, self).__init__()

        freeze_clip, _, _ = create_model_and_transforms('ViT-B-16', image_size,
                                                        pretrained='openai')
        self.model = freeze_clip
        self.feature_list = [4, 8, 12]

    def forward(self, x):
        _, patch_tokens = self.model.encode_image(x, self.feature_list)

        pure_patch_tokens = [s[:, 1:, :] for s in patch_tokens]

        B, L, C = pure_patch_tokens[0].shape
        H = int(np.sqrt(L))
        patch_map = [s.permute(0,2,1).view(B, C, H, H) for s in pure_patch_tokens]

        return patch_map
