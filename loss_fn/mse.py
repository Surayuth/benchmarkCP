import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        """
        Mean squared error
        """
        super(MSELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(-1)
        return F.mse_loss(
            y_pred, 
            y_true, 
            weight=self.weight,
            reduction=self.reduction)
    