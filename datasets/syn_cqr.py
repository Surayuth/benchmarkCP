import torch
from torch.utils.data import Dataset

def gen_xy(n, seed=42):
    """
    Synthetic dataset according to 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    g = torch.Generator()
    g.manual_seed(seed)

    x = 5 * torch.rand(n, generator=g)
    y = torch.poisson(torch.sin(x).pow(2) + 0.1)

    e1 = torch.randn(n, generator=g)
    y += 0.03 * x * e1

    u = torch.rand(n, generator=g)
    e2 = torch.randn(n, generator=g)
    y += 25 * (u < 0.01) * e2

    return x, y

class SynCQR(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(1)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] 

def gen_syn_data(val_size=-1, calib_size=-1):
    """
    val_size and calib_szie for compatibility
    """

    N_tr=2000
    N_va=1000
    N_ca=1000
    N_te=5000

    X_tr, y_tr = gen_xy(N_tr)
    X_va, y_va = gen_xy(N_va)
    X_ca, y_ca = gen_xy(N_ca)
    X_te, y_te = gen_xy(N_te)

    train_dataset = SynCQR(X_tr, y_tr)
    val_dataset = SynCQR(X_va, y_va)
    calib_dataset = SynCQR(X_ca, y_ca)
    test_dataset = SynCQR(X_te, y_te)
    
    # For compatibility with the typical classification dataset
    class_dict = {}
    return class_dict, train_dataset, val_dataset, calib_dataset, test_dataset



