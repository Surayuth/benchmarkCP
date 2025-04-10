import torch
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseCP(ABC):
    def __init__(self, method_args, device, net, alpha, n_classes):
        self.method_args = method_args
        self.device = device
        self.net = net
        self.alpha = alpha
        self.n_classes = n_classes
        self.qhat = None
        self.cond_qhats = None
    
    def calculate_scores(self, calib_loader):
        """
        Calculate nonconformity scores from calibration set
        """
        scores = []

        tot_targets = []
        for batch_idx, (inputs, targets) in tqdm(enumerate(calib_loader), 
                                                 total=len(calib_loader), 
                                                 desc="Calculating non-conformity scores",
                                                 bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'
                                                 ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                tot_targets += targets.cpu().tolist()

                outputs = self.net(inputs)

                probs = F.softmax(outputs, dim=1)
                batch_scores = self.score_func(probs, targets)
                scores += batch_scores.cpu().tolist()

        tot_targets = np.array(tot_targets)

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
    
    def calculate_qhat(self, calib_loader):
        scores, cond_scores, cond_samples = self.calculate_scores(calib_loader)

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
        self.cond_qhats = cond_qhats

    @abstractmethod
    def calculate_pred_set(self, outputs, normalized=False):
        pass
    
    @abstractmethod
    def score_func(self, probs, targets, **kwargs):
        pass