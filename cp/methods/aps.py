# paper: https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf 
import torch
import torch.nn.functional as F
from cp import BaseCP


class APSCP(BaseCP):
    def __init__(self, method_args, device, net, alpha, n_classes):
        super().__init__(method_args, device, net, alpha, n_classes)
    
    def score_func(self, outputs, targets):
        """
        Calculate batch nonconformity scores.
        """
        probs = F.softmax(outputs, dim=1)

        N = probs.shape[0]

        # sort prob and target 
        sorted_p, sorted_idx = probs.sort(descending=True, dim=1)
        target_idx = torch.where(sorted_idx == targets.unsqueeze(1))[1]

        # calculate p at target
        p_at_target = sorted_p[torch.arange(N), target_idx]

        # calculate cumsum p to target
        cumsum_p = sorted_p.cumsum(dim=1)
        cumsum_p_to_target = cumsum_p[torch.arange(N), target_idx]

        # calculate E
        # we use this formula because we need the target in the prediction set precisely
        # this requires u > V. After solving the equation, we get the batch_scores.
        u = torch.rand(N, device=self.device)
        batch_scores = cumsum_p_to_target - u * p_at_target

        return batch_scores

    def calculate_pred_set(self, outputs):
        """
        Calculate pred set for marginal and class-cond types.
        Output: pred_sets (marginal pred sets, cls-cond pred sets)
        """
        probs = F.softmax(outputs, dim=1)

        N, C = probs.shape

        # calculate cumsum p and prob at last idx
        sorted_p, sorted_idx = probs.sort(descending=True, dim=1)
        cumsum_p = sorted_p.cumsum(dim=1)

        pred_sets = []
        u = torch.rand(N, device=self.device)
        for cp_type in ["marginal", "class-cond"]:
            # this method only changes qhat for marg and class-cond (sec. 2.5)
            if cp_type == "marginal":
                qhat = self.qhat
            elif cp_type == "class-cond":
                qhat = max(self.cond_qhats)

            last_included_idx = (cumsum_p >= qhat).int().argmax(dim=1)
            cumsum_p_to_idx = cumsum_p[torch.arange(N), last_included_idx]
            prob_at_idx = sorted_p[torch.arange(N), last_included_idx]

            # compare u <= v
            v = 1 / prob_at_idx * (cumsum_p_to_idx - qhat)
            final_included_idx = last_included_idx - (u <= v).int()
            selected_idx = torch.arange(C, device=self.device).expand(N, -1) <= final_included_idx.unsqueeze(1)

            # we need to rearrange each idx to the correct position
            back_idx = sorted_idx.argsort(dim=1)
            _pred_sets = selected_idx.gather(1, back_idx)

            pred_sets.append(_pred_sets)

        return pred_sets 
            

        
