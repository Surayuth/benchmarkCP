from abc import ABC, abstractmethod

class BaseCP(ABC):
    def __init__(self, device, net, alpha, n_classes, calib_loader):
        self.device = device
        self.net = net
        self.alpha = alpha
        self.n_classes = n_classes
        self.calib_loader = calib_loader
        self.qhat = None
        self.cond_qhats = None
    
    @abstractmethod
    def calculate_scores(self):
        pass
    
    @abstractmethod
    def calculate_qhat(self):
        pass
    
    @abstractmethod
    def calculate_pred_set(self):
        pass