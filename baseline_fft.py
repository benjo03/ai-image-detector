"""
Optional (Day 2): a CLASSICAL computer-vision baseline - no deep learning.

GAN-generated images leave traces in the frequency domain: the upsampling layers
of the generator distort the high-frequency part of the image spectrum.
We compute the 2D Fourier transform of each image, average it in rings around the
centre (-> a 1D 'spectrum profile'), and train logistic regression on that.

This gives you a baseline to compare the CNN against, and a very interpretable plot.
Run:  python baseline_fft.py
Creates outputs/fft_spectrum.png and outputs/baseline_metrics.json
"""
import json
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from src.data import find_data_dir

SIZE = 256
N_TRAIN = 4000   # images used to fit the baseline (it's fast, but reading files takes time)
N_TEST = 2000

# Pre-compute the distance of every pixel from the spectrum centre (used for ring averages)
_yy, _xx = np.indices((SIZE, SIZE))
_RADIUS = np.hypot(_xx - SIZE // 2, _yy - SIZE // 2).astype(int).ravel()
_COUNTS = np.bincount(_RADIUS)


def spectrum_profile(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE, SIZE)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(img))            # low frequencies in the centre
    magnitude = 20 * np.log(np.abs(spectrum) + 1e-8)          # log scale (like decibels)
    profile = np.bincount(_RADIUS, magnitude.ravel()) / _COUNTS  # average per ring
    profile = profile[: SIZE // 2]                            # radius 0 .. max frequency
    return (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)


def load_split(split, n):
    root = find_data_dir() / config.SPLITS[split]
    files = [(p, 1) for p in (root / "fake").iterdir()] + [(p, 0) for p in (root / "real").iterdir()]
    random.Random(config.SEED).shuffle(files)
    files = files[:n]
    X = np.stack([spectrum_profile(p) for p, _ in tqdm(files, desc=f"FFT {split}")])
    y = np.array([label for _, label in files])
    return X, y


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_split("train", N_TRAIN)
    X_test, y_test = load_split("test", N_TEST)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_fake": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan"),
        "n_test_images": int(len(y_test)),
    }
    print("FFT + logistic regression baseline:")
    for k, v in metrics.items():
        print(f"{k:>15}: {v}")
    (config.OUTPUT_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Interpretable plot: average spectrum profile of real vs fake images
    fig, ax = plt.subplots(figsize=(7, 4))
    freqs = np.arange(X_train.shape[1])
    for label, name in [(0, "real"), (1, "fake")]:
        rows = X_train[y_train == label]
        mean, std = rows.mean(0), rows.std(0)
        ax.plot(freqs, mean, label=name)
        ax.fill_between(freqs, mean - std, mean + std, alpha=0.2)
    ax.set(xlabel="spatial frequency (low -> high)", ylabel="normalized log power",
           title="Average frequency spectrum: real vs AI-generated")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "fft_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"Saved fft_spectrum.png and baseline_metrics.json in {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
