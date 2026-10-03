"""
Grad-CAM, implemented from scratch (about 30 lines) so you can explain every step.

Idea: take the last conv layer's feature maps (A) and the gradient of the
model's score with respect to them (dScore/dA). Average the gradient over each
feature map -> one importance weight per channel. Weighted sum of the maps,
ReLU, upscale to image size -> heatmap of 'where the model looked'.
"""
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inp, out):
        self.activations = out
        out.register_hook(self._save_gradients)  # called during backward()

    def _save_gradients(self, grad):
        self.gradients = grad

    def __call__(self, x):
        """x: one image tensor of shape (1, 3, H, W).
        Returns (heatmap HxW in [0,1], P(fake))."""
        self.model.eval()
        self.model.zero_grad()
        logit = self.model(x).squeeze()
        prob_fake = torch.sigmoid(logit).item()

        # Explain the class the model actually predicted:
        # evidence for FAKE = +logit, evidence for REAL = -logit
        score = logit if prob_fake >= 0.5 else -logit
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)           # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam, prob_fake


def overlay(img, cam, alpha=0.45):
    """img: HxWx3 in [0,1], cam: HxW in [0,1] -> blended HxWx3 image."""
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)
