# paper: https://arxiv.org/pdf/2009.14193
import torch
import numpy as np
import torch.nn.functional as F
from cp import BaseCP

class RAPSCP(BaseCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def score_func(self, probs, targets, lamb, kreg):
        pass

    def calculate_pred_set(self, outputs, normalized=False):
        pass
