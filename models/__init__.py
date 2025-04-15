from .cifar10_resnet import load_cifar10_resnet
from .imagenet1k_resnet import load_imagenet1k_resnet
from .simple_reg_net import load_simple_reg_net
from .hydra_reg import load_hydra_reg_net

# TODO: add model args like in cp methods

_MODELS = {
    "cifar10_resnet": load_cifar10_resnet,
    "imagenet1k_resnet": load_imagenet1k_resnet,
    'simple_reg_net': load_simple_reg_net,
    'hydra_reg': load_hydra_reg_net
}

def load_model(model_name, version):
    if model_name not in _MODELS:
        available = list(_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return _MODELS[model_name](version)