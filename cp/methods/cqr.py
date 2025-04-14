import torch
import torch.nn.functional as F
from cp import BaseCP

class CQR(BaseCP):
    def __init__(self, method_args, device, net, alpha, n_classes):
        super().__init__(method_args, device, net, alpha, n_classes)

    def score_fun(self, outputs, targets):
        pass

    def calculate_pred_set(self, outputs):
        pass