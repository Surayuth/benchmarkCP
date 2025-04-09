from .cifar10_resnet import load_cifar10_resnet
from .imagenet1k_resnet import load_imagenet1k_resnet

# TODO: add model args like in cp methods

_MODELS = {
    "cifar10_resnet": load_cifar10_resnet,
    "imagenet1k_resnet": load_imagenet1k_resnet
}

def load_model(model_name, version):
    if model_name not in _MODELS:
        available = list(_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return _MODELS[model_name](version)