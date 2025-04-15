import torch
import torch.nn.functional as F
from cp import BaseCP

class CQR(BaseCP):
    def __init__(self, method_args, device, net, alpha, n_classes):
        super().__init__(method_args, device, net, alpha, n_classes)

    def score_func(self, outputs, targets):
        ql = outputs[:, 0]
        qh = outputs[:, 2]
        e = torch.maximum(ql - targets, targets - qh)
        return e

    def calculate_pred_set(self, outputs):
        d = torch.tensor([-self.qhat, self.qhat], device=self.device).unsqueeze(0)

        ql = outputs[:, 0].unsqueeze(1)
        qh = outputs[:, 2].unsqueeze(1)
        
        pred_set = torch.cat((ql, qh), dim=1) + d

        return pred_set