# see: https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf 
import torch
import numpy as np
import torch.nn.functional as F
from cp import BaseCP


class APSCP(BaseCP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_scores(self):
        scores = []

        tot_targets = []
        for batch_idx, (inputs, targets) in enumerate(self.calib_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                tot_targets += targets.cpu().tolist()

                outputs = self.net(inputs)

                probs = F.softmax(outputs, dim=1)

                # calculate E (scores)
                u = torch.rand(targets.shape[0])

                y = F.one_hot(targets, num_classes=probs.shape[1])
                sorted_idx = torch.argsort(probs, descending=True, dim=1)
                sorted_p = torch.gather(probs, 1, sorted_idx)
                sorted_y = torch.gather(y, 1, sorted_idx)
                sorted_target_idx = torch.argmax(sorted_y, dim=1)
                cumsum_sorted_p_to_target = self.calculate_cumsum_to_target(sorted_p, sorted_target_idx)
                batch_scores = (cumsum_sorted_p_to_target - u * sorted_p[torch.arange(probs.shape[0]), sorted_target_idx]).cpu().tolist()

                scores += batch_scores

        # marginal scores
        scores = np.array(scores)

        # cond scores
        cond_scores = []
        cond_samples = []
        for c in range(self.n_classes):
            cls_idx = np.where(tot_targets == c)[0]
            cond_scores.append(scores[cls_idx])
            cond_samples.append(len(cls_idx))

        return scores, cond_scores, cond_samples
             
    def calculate_cumsum_to_target(self, sorted_p, sorted_target_idx):
            N, C = sorted_p.shape

            # Create position indices for each element in the batch
            positions = torch.arange(C).unsqueeze(0).expand(N, -1)
            positions = positions.to(self.device)

            # Create mask for positions up to and including target position
            mask = positions <= sorted_target_idx.unsqueeze(1)

            # Apply mask and sum
            cumsum_sorted_p_to_target = (sorted_p * mask).sum(dim=1)

            return cumsum_sorted_p_to_target

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
   
        # Calculate last included index
        sorted_idx = torch.argsort(probs, descending=True, dim=1)
        sorted_p = torch.gather(probs, 1, sorted_idx)
        cumsum_p = torch.cumsum(sorted_p, dim=1)
        last_included_idx = torch.argmax((cumsum_p + 1e-8 >= self.qhat).float(), axis=1)

        # Calculate V
        cumsum_to_last_index = cumsum_p[torch.arange(len(probs)), last_included_idx]
        prob_last_idx = sorted_p[torch.arange(len(probs)), last_included_idx]
        V = 1 / prob_last_idx * (cumsum_to_last_index - self.qhat)

        # Calculate S (pred_set)
        U = torch.rand(len(probs)).to(self.device)
        # exclude last index if U <= V
        final_included_idx = last_included_idx - (U <= V) * 1
        N, C = probs.shape
        # selected_idx is based on the sorted_idx
        # So, we need to reverse them back to orignal class (by sorting again)
        selected_idx = torch.arange(C, device=self.device).expand(N, -1) <= final_included_idx.unsqueeze(1)
        pred_sets = torch.gather(selected_idx, 1, sorted_idx)

        # TODO: implement cond_pred_sets
        return pred_sets #, cond_pred_sets

        
