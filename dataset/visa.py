import json
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from .base import BaseDataset
'''dataset source: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'''


VISA_CLS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum',
]

from config import DATA_ROOT

VISA_ROOT = os.path.join(DATA_ROOT, 'VisA_20220922')

class VisaDataset(BaseDataset):
    def __init__(self, class_names, transform, target_transform, synthesis_anomalies=True,
                 root=VISA_ROOT, training=True,white_noise=False,
                 mode='FS',
                 k_shot=5):
        super(VisaDataset, self).__init__(class_names=class_names,
                                          transform=transform,
                                          target_transform=target_transform,
                                          root=root, training=training,white_noise=white_noise,
                                          mode=mode, k_shot=k_shot,synthesis_anomalies=synthesis_anomalies)

