import torch
import numpy as np
import torch.nn.functional as F
from cp import BaseCP


class HingeCP(BaseCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_scores(self):
        tot_targets = []
        cond_scores = []
        cond_samples = []

        true_probs = []
        for batch_idx, (inputs, targets) in enumerate(self.calib_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                tot_targets += targets.cpu().tolist()

                probs = F.softmax(self.net(inputs), dim=1)
                indices = torch.arange(probs.size(0), device=probs.device)
                _true_probs = probs[indices, targets].cpu().tolist()
                true_probs += _true_probs

        tot_targets = np.array(tot_targets)
        true_probs = np.array(true_probs)

        # marginal scores
        scores = 1 - true_probs

        # cond scores
        for c in range(self.n_classes):
            cls_idx = np.where(tot_targets == c)[0]
            cls_probs = true_probs[cls_idx]
            cond_scores.append(1 - cls_probs)
            cond_samples.append(len(cls_idx))

        return scores, cond_scores, cond_samples
    
    def calculate_qhat(self):
        scores, cond_scores, cond_samples = self.calculate_scores()

        # marginal qhat
        n_calib = len(scores)
        q = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        self.qhat = np.quantile(scores, q, method="higher")

        # cond qhat
        cond_qhats = []
        for c in range(self.n_classes):
            n_calib = cond_samples[c]
            q = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
            cond_qhat = np.quantile(cond_scores[c], q, method="higher")
            cond_qhats.append(cond_qhat)
        self.cond_qhats = torch.tensor(cond_qhats).to(self.device)

    def calculate_pred_set(self, outputs, normalized=False):
        if normalized:
            probs = outputs
        else:
            probs = F.softmax(outputs, dim=1)
        
        # calculate marginal pred set
        pred_sets = probs >= (1 - self.qhat)

        # calculate cond pred_size
        cond_pred_sets = probs >= (1 - self.cond_qhats)

        return pred_sets, cond_pred_sets