import json
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from .base import BaseDataset

'''dataset source: https://www.robots.ox.ac.uk/~vgg/data/dtd/'''

DTD_CLS_NAMES = [
    'Blotchy_099', 'Fibrous_183', 'Marbled_078', 'Matted_069', 'Mesh_114','Perforated_037','Stratified_154','Woven_001','Woven_068','Woven_104','Woven_125','Woven_127',
]
from config import DATA_ROOT

DTD_ROOT = os.path.join(DATA_ROOT, 'DTD-Synthetic')

class DTDDataset(BaseDataset):
    def __init__(self, class_names, transform, target_transform, synthesis_anomalies, root=DTD_ROOT, training=True,
                 mode='FS',white_noise=False,
                 k_shot=5):
        super(DTDDataset, self).__init__(class_names=class_names,
                  transform=transform,
                  target_transform=target_transform,
                  root=root, training=training,white_noise=white_noise,
                  mode=mode, k_shot=k_shot, synthesis_anomalies=synthesis_anomalies)

