import json
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from .base import BaseDataset

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''


MVTEC_CLS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]
from config import DATA_ROOT

MVTEC_ROOT = os.path.join(DATA_ROOT, 'mvtec_anomaly_detection')

class MVTecDataset(BaseDataset):
    def __init__(self, class_names, transform, target_transform,
                 synthesis_anomalies=False, root=MVTEC_ROOT, training=True, mode='FS', k_shot=5,white_noise=False):
        super(MVTecDataset, self).__init__(class_names=class_names,
                                           transform=transform,
                                           target_transform=target_transform,
                                           root=root, training=training,white_noise=white_noise,
                                           mode=mode, k_shot=k_shot, synthesis_anomalies=synthesis_anomalies)

