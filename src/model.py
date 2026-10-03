"""
The CNN. We take a network pretrained on ImageNet and replace its last layer
with a single output: the 'fake score' (a logit; sigmoid(logit) = P(fake)).
"""
import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(name=None, pretrained=None):
    name = name or config.MODEL_NAME
    pretrained = config.PRETRAINED if pretrained is None else pretrained

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model


def get_target_layer(model, name=None):
    """Last convolutional block - this is where Grad-CAM looks."""
    name = name or config.MODEL_NAME
    if name == "resnet18":
        return model.layer4[-1]
    return model.features[-1]


def extract_features(model, x, name=None):
    """Feature vector BEFORE the final classification layer (512 numbers for ResNet18).
    Not used for the image project itself - this is the hook for Phase 2 (video):
    run every video frame through this and feed the sequence to an LSTM/GRU."""
    name = name or config.MODEL_NAME
    if name == "resnet18":
        body = nn.Sequential(*list(model.children())[:-1])  # everything except fc
        return torch.flatten(body(x), 1)
    feats = model.avgpool(model.features(x))
    return torch.flatten(feats, 1)


def save_checkpoint(model, path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": config.MODEL_NAME, "state_dict": model.state_dict(), **extra}, path)


def load_model(path, device):
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run train.py first.")
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
