from transformers import ViTModel, ViTMAEModel
from torch import nn
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from itertools import repeat
import collections.abc

from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic'):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('embeddings.position_embeddings', None)
    grid_size = (model.config.image_size // model.config.patch_size, model.config.image_size // model.config.patch_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    old_pos_embed = old_pos_embed.squeeze(0)
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    new_pos_embed = new_pos_embed.unsqueeze(0)
    state_dict['embeddings.position_embeddings'] = new_pos_embed


class ViT_backbone(nn.Module):
    def __init__(self, backbone, image_size):
        super(ViT_backbone, self).__init__()

        if backbone == 'VIT':
            model_name = 'google/vit-base-patch16-224'
            model_pre = ViTModel.from_pretrained(model_name, output_hidden_states=True)
            config = model_pre.config
            config.image_size = image_size
            self.model = ViTModel(config)
            self.layer_nums = [4-1, 8-1 ,12-1]
        elif backbone == 'MAE':
            model_name = 'facebook/vit-mae-base'
            self.layer_nums = [4-1, 8-1, 12-1]
            model_pre = ViTMAEModel.from_pretrained(model_name, output_hidden_states=True)
            config = model_pre.config
            config.image_size = image_size
            self.model = ViTMAEModel(config)
        else:
            raise NotImplementedError

        state_dict = model_pre.state_dict()
        resize_pos_embed(state_dict, self.model)
        incompatible_keys = self.model.load_state_dict(state_dict, strict=True)


    def forward(self, x):
        outputs = self.model(x)
        hidden_states = outputs.hidden_states
        selected_layers = [hidden_states[i] for i in self.layer_nums]

        pure_patch_tokens = [s[:, 1:, :] for s in selected_layers]

        B, L, C = pure_patch_tokens[0].shape
        H = int(np.sqrt(L))
        patch_map = [s.permute(0,2,1).view(B, C, H, H) for s in pure_patch_tokens]

        # patch_map = [F.normalize(s, dim=1) for s in patch_map]

        return patch_map



