import torch
import numpy as np
import torch.nn.functional as F
from cp import BaseCP


class HingeCP(BaseCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def score_func(self, probs, targets, *args, **kwargs):
        target_probs = probs[torch.arange(len(probs)), targets]
        return 1 - target_probs