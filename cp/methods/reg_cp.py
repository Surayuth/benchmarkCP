import torch
import torch.nn.functional as F
from cp import BaseCP

class RegCP(BaseCP):
    def __init__(self, method_args, device, net, alpha, n_classes):
        super().__init__(method_args, device, net, alpha, n_classes)

    def score_func(self, outputs, targets):
        outputs = outputs.reshape(-1)
        r = torch.abs(targets - outputs.reshape(-1))
        return r

    def calculate_pred_set(self, outputs):
        d = torch.tensor([-self.qhat, self.qhat], device=self.device).unsqueeze(0)
        pred_set = outputs.expand(len(outputs), 2) + d
        return pred_set