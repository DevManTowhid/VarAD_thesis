import json
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from .base import BaseDataset

'''dataset source: https://avires.dimi.uniud.it/papers/btad/btad.zip'''
BTAD_CLS_NAMES = [
    '01', '02', '03',
]

from config import DATA_ROOT

BTAD_ROOT = os.path.join(DATA_ROOT, 'BTech_Dataset_transformed')


class BTADDataset(BaseDataset):
    def __init__(self, class_names, transform, target_transform, synthesis_anomalies, root=BTAD_ROOT, training=True,
                 mode='FS',white_noise=False,
                 k_shot=5):
        super(BTADDataset, self).__init__(class_names=class_names,
                  transform=transform,
                  target_transform=target_transform, root=root, training=training,white_noise=white_noise,
                  mode=mode, k_shot=k_shot, synthesis_anomalies=synthesis_anomalies)

