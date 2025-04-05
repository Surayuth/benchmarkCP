from .resnet import load_resnet

_MODELS = {
    "resnet": load_resnet
}

def load_model(model_name, version):
    if model_name not in _MODELS:
        available = list(_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return _MODELS[model_name](version)