"""
Loading the images and turning them into tensors the model can use.

Label convention used everywhere in this project:
    1 = FAKE (AI-generated)
    0 = REAL
"""
import random
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config

# ImageNet statistics - required because the backbone was pretrained on ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train):
    if train:
        # Light augmentation: makes the model more robust, avoids overfitting
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class FakeIsOne:
    """ImageFolder numbers classes alphabetically (fake=0, real=1).
    We flip that so 1 means 'fake', which is the class we want to detect."""

    def __init__(self, fake_idx):
        self.fake_idx = fake_idx

    def __call__(self, y):
        return 1 if y == self.fake_idx else 0


@lru_cache(maxsize=None)
def find_data_dir():
    """Return the folder containing train/valid/test. Tries config.DATA_DIR first,
    then searches inside ./data for a train/ folder that has real/ and fake/."""
    if (config.DATA_DIR / config.SPLITS["train"]).exists():
        return config.DATA_DIR
    data_root = Path("data")
    if data_root.exists():
        for p in data_root.rglob(config.SPLITS["train"]):
            if (p / "real").is_dir() and (p / "fake").is_dir():
                print(f"[data] Found dataset at: {p.parent}")
                return p.parent
    raise FileNotFoundError(
        "Could not find the dataset.\n"
        "Expected a folder like:\n"
        "   data/.../train/real, data/.../train/fake\n"
        "   data/.../valid/real, data/.../valid/fake\n"
        "   data/.../test/real,  data/.../test/fake\n"
        "Download it (see GUIDE.md, Day 1) or set DATA_DIR in config.py."
    )


def make_dataset(split, train=False, subset=None):
    root = find_data_dir() / config.SPLITS[split]
    ds = datasets.ImageFolder(root, transform=get_transforms(train))
    ds.target_transform = FakeIsOne(ds.class_to_idx["fake"])
    if subset is not None and subset < len(ds):
        rng = random.Random(config.SEED)
        ds = Subset(ds, rng.sample(range(len(ds)), subset))
    return ds


def get_loader(split):
    subsets = {"train": config.TRAIN_SUBSET, "val": config.VAL_SUBSET, "test": config.TEST_SUBSET}
    is_train = split == "train"
    ds = make_dataset(split, train=is_train, subset=subsets[split])
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=is_train,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def denormalize(t):
    """Undo Normalize so an image tensor can be displayed. Returns HxWx3 numpy in [0,1]."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = (t.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
