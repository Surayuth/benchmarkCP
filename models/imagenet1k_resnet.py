from torchvision.models import resnet18, ResNet18_Weights

def load_imagenet1k_resnet(version):
    "source: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18"
    if version == "18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model