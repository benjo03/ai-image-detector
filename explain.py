"""
Step 4: interpretability - WHY does the model say real or fake?
Run:  python explain.py
Creates in outputs/:
  gradcam_examples.png  - heatmaps for correct AND wrong predictions
  gradcam_average.png   - average heatmap over many real vs many fake images
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import config
from src.data import denormalize, make_dataset
from src.gradcam import GradCAM, overlay
from src.model import get_target_layer, load_model
from src.utils import get_device

PER_GROUP = 4        # images shown per group in the examples figure
N_AVERAGE = 300      # images per class used for the average heatmap
GROUPS = {
    "Correct: FAKE": (1, 1),   # (true label, predicted label)
    "Correct: REAL": (0, 0),
    "Wrong: fake called real": (1, 0),
    "Wrong: real called fake": (0, 1),
}


def main():
    device = get_device()
    model = load_model(config.CHECKPOINT_PATH, device)
    cam = GradCAM(model, get_target_layer(model))
    ds = make_dataset("test", train=False, subset=config.TEST_SUBSET)

    picked = {g: [] for g in GROUPS}
    avg = {0: [], 1: []}

    for i in tqdm(range(len(ds)), desc="grad-cam"):
        need_examples = any(len(v) < PER_GROUP for v in picked.values())
        need_average = any(len(v) < N_AVERAGE for v in avg.values())
        if not (need_examples or need_average):
            break

        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        heat, p_fake = cam(x)
        pred = int(p_fake >= 0.5)

        if len(avg[y]) < N_AVERAGE:
            avg[y].append(heat)
        for name, (t, p) in GROUPS.items():
            if (y, pred) == (t, p) and len(picked[name]) < PER_GROUP:
                picked[name].append((denormalize(x[0]), heat, p_fake))

    # ---- Figure 1: example heatmaps ----
    fig, axes = plt.subplots(len(GROUPS), PER_GROUP * 2, figsize=(PER_GROUP * 4, len(GROUPS) * 2.3))
    for r, (name, items) in enumerate(picked.items()):
        for c in range(PER_GROUP):
            ax_img, ax_cam = axes[r, 2 * c], axes[r, 2 * c + 1]
            ax_img.axis("off")
            ax_cam.axis("off")
            if c < len(items):
                img, heat, p = items[c]
                ax_img.imshow(img)
                ax_cam.imshow(overlay(img, heat))
                ax_cam.set_title(f"P(fake)={p:.2f}", fontsize=8)
        axes[r, 0].set_title(name, fontsize=9, loc="left", fontweight="bold")
    fig.suptitle("Grad-CAM: where the model looks (red = most important)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "gradcam_examples.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: average heatmap per class ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (label, name) in zip(axes, [(0, "REAL images"), (1, "FAKE images")]):
        if avg[label]:
            ax.imshow(np.mean(avg[label], axis=0), cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"Average Grad-CAM, {name}\n(n={len(avg[label])})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "gradcam_average.png", dpi=150)
    plt.close(fig)

    for name, items in picked.items():
        print(f"{name}: {len(items)} examples found")
    print(f"Saved gradcam_examples.png and gradcam_average.png in {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
