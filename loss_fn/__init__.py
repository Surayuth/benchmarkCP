from .ce import CELoss
from .mse import MSELoss
from .cqr_loss import CQRLoss

_LOSSFN = {
    'ce': CELoss,
    'mse': MSELoss,
    'cqr_loss': CQRLoss
}

_LOSSFN = {
    'ce': {
        'class': CELoss,
        'params': []
    },
    'mse': {
        'class': MSELoss,
        'params': []
    },
    'cqr_loss': {
        'class': CQRLoss,
        'params': [
            {'name': 'loss_alpha', 'type': float, 'default': 0.05, 'help': 'Do not need to be equal to alpha (as the qhat will correct it anyway).'},
            # Note in page 6 from the paper (https://arxiv.org/pdf/1905.03222).
            # Quantile regression is sometimes too conservative, resulting in unnecessarily wide prediction
            # intervals. In our experience, quantile regression forests [22] are often overly conservative
            # and quantile neural networks [20] are occasionally so. We can mitigate this problem by
            # tuning the nominal quantiles of the underlying method as additional hyper-parameters in
            # cross validation. Notably, this tuning does not invalidate the coverage guarantee, but it may
            # yield shorter intervals, as our experiments confirm.

            # (My comment) In other words, CQR will coverage the range to ensure correct coverage anyway.
            # So, if we would like to obtain short pred size, we could try to find tune the nominal quantile (loss_alpha).
            # Note that nominal quantile (alpha_low, alpha_high) is the concept of quantile regression while the expected coverage (alpha)
            # is the concept from conformal prediciton meaning the miscoverage rate.
        ]
    }
}

def get_loss_fn(loss_fn_name, loss_args):
    if loss_fn_name not in _LOSSFN:
        available = list(_LOSSFN.keys())
        raise ValueError(f"Loss function '{loss_fn_name}' not found. Available loss functions: {available}")

    return _LOSSFN[loss_fn_name]['class'](**loss_args)
