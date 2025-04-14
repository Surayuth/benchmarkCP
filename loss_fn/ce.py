import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        """
        Cross entropy loss
        """
        super(CELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return F.cross_entropy(
            y_pred, 
            y_true, 
            weight=self.weight,
            reduction=self.reduction)