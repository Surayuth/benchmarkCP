import torch
import torch.nn as nn
import torch.nn.functional as F

class CQRLoss(nn.Module):
    def __init__(self, loss_alpha=0.1):
        super(CQRLoss, self).__init__()
        self.al = loss_alpha / 2
        self.ah = 1 - loss_alpha / 2

    def forward(self, y_pred, y_true):
        """
        shape of y_pred must be (N, 3)
        """
        if y_pred.shape[1] == 3:
            pl = y_pred[:, 0]
            pm = y_pred[:, 1]
            ph = y_pred[:, 2]

            loss_l = torch.where(y_true > pl, self.al * (y_true - pl), (1 - self.al) * (pl - y_true)).mean()
            loss_m = F.mse_loss(pm, y_true, reduction="mean")
            loss_h = torch.where(y_true > ph, self.ah * (y_true - ph), (1 - self.ah) * (ph - y_true)).mean()

            return loss_l + loss_m + loss_h
        else:
            raise Exception("The input dims must be 3!")