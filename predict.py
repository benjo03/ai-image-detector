"""
Step 5: try the model on ANY image (e.g. your own photo, or an AI-generated face).
Run:  python predict.py path/to/image.jpg
      python predict.py path/to/image.jpg --no-face-crop
Saves outputs/prediction_<image name>.png with the image + Grad-CAM heatmap.

The model was trained on face crops, so by default we first find the face
with OpenCV's built-in face detector and crop it (classical computer vision).
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from src.data import denormalize, get_transforms
from src.gradcam import GradCAM, overlay
from src.model import get_target_layer, load_model
from src.utils import get_device


def crop_largest_face(pil_img, margin=0.35):
    """Detect faces with a Haar cascade and return the largest one (with some margin).
    Returns the original image if no face is found."""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        print("No face detected - using the whole image.")
        return pil_img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(margin * max(w, h))
    W, H = pil_img.size
    box = (max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m))
    return pil_img.crop(box)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()

    device = get_device()
    model = load_model(config.CHECKPOINT_PATH, device)
    cam = GradCAM(model, get_target_layer(model))

    img = Image.open(args.image).convert("RGB")
    if not args.no_face_crop:
        img = crop_largest_face(img)

    x = get_transforms(train=False)(img).unsqueeze(0).to(device)
    heat, p_fake = cam(x)
    label = "AI-GENERATED (fake)" if p_fake >= 0.5 else "REAL"
    confidence = p_fake if p_fake >= 0.5 else 1 - p_fake
    print(f"Prediction: {label}  (confidence {confidence:.1%}, P(fake) = {p_fake:.3f})")

    shown = denormalize(x[0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    axes[0].imshow(shown)
    axes[0].set_title("Input (face crop)")
    axes[1].imshow(overlay(shown, heat))
    axes[1].set_title(f"{label}\nconfidence {confidence:.1%}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"prediction_{args.image.stem}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
