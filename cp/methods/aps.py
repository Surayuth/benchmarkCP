# see: https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf 
import torch
import numpy as np
import torch.nn.functional as F
from cp import BaseCP


class APSCP(BaseCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def score_func(self, probs, targets, *args, **kwargs):
        """
        Calculate batch nonconformity scores.
        """
        y = F.one_hot(targets, num_classes=probs.shape[1])

        # sort prob and target 
        sorted_idx = torch.argsort(probs, descending=True, dim=1)
        sorted_p = torch.gather(probs, 1, sorted_idx)
        sorted_y = torch.gather(y, 1, sorted_idx)
        sorted_target_idx = torch.argmax(sorted_y, dim=1)

        # calculate p at target
        p_at_target = sorted_p[torch.arange(len(probs)), sorted_target_idx]

        # calculate cumsum p to target
        cumsum_p = torch.cumsum(sorted_p, dim=1)
        cumsum_p_at_target = cumsum_p[torch.arange(len(probs)), sorted_target_idx]

        # calculate E
        # we use this formula because we need the target in the prediction set precisely
        # this requires u > V. After solving the equation, we get the batch_scores.
        u = torch.rand(len(probs), device=self.device)
        batch_scores = cumsum_p_at_target - u * p_at_target

        return batch_scores
            

        
