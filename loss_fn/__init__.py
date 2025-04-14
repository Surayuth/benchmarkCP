from .ce import CELoss
from .mse import MSELoss

_LOSSFN = {
    'ce': CELoss,
    'mse': MSELoss,
}

def get_loss_fn(loss_fn_name, **kwargs):
    if loss_fn_name not in _LOSSFN:
        available = list(_LOSSFN.keys())
        raise ValueError(f"Loss function '{loss_fn_name}' not found. Available loss functions: {available}")
    
    return _LOSSFN[loss_fn_name](**kwargs)
