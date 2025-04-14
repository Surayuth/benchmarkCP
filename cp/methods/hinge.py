import torch
import numpy as np
import torch.nn.functional as F
from cp import BaseCP


class HingeCP(BaseCP):
    def __init__(self, method_args, device, net, alpha, n_classes):
        super().__init__(method_args, device, net, alpha, n_classes)

    def score_func(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        target_probs = probs[torch.arange(len(probs)), targets]
        return 1 - target_probs

    def calculate_pred_set(self, outputs):
        probs = F.softmax(outputs, dim=1)

        pred_sets = torch.zeros(probs.shape, dtype=torch.bool, device=self.device)
        cond_pred_sets = torch.zeros(probs.shape, dtype=torch.bool, device=self.device)

        for c in range(self.n_classes):
            c_targets = c * torch.ones(len(probs), dtype=torch.long, device=self.device)

            c_scores = self.score_func(probs, c_targets)

            pred_sets[:, c] = c_scores <= self.qhat
            cond_pred_sets[:, c] = c_scores <= self.cond_qhats[c]

        return pred_sets, cond_pred_sets