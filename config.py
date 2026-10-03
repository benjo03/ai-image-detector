"""
All settings for the project live here, so you only ever edit one file.
"""
from pathlib import Path

# ---------------- Paths ----------------
# Folder that contains train/, valid/ and test/, each with real/ and fake/ inside.
# For the Kaggle "140k Real and Fake Faces" dataset this is the default below.
# If your folder is somewhere else, change this line (the code will also try to find it automatically).
DATA_DIR = Path("data/real_vs_fake/real-vs-fake")
SPLITS = {"train": "train", "val": "valid", "test": "test"}

OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"

# ---------------- Data ----------------
IMG_SIZE = 224          # input size for the CNN
# How many images to use from each split. Smaller = faster training.
# Set to None to use everything (100k train images - only do this on a GPU).
TRAIN_SUBSET = 20000
VAL_SUBSET = 4000
TEST_SUBSET = 4000
BATCH_SIZE = 64         # lower to 32 or 16 if you get "out of memory" errors
NUM_WORKERS = 2         # set to 0 if you get DataLoader errors on Windows

# ---------------- Model / training ----------------
MODEL_NAME = "resnet18"  # "resnet18" or "efficientnet_b0"
PRETRAINED = True        # start from ImageNet weights (transfer learning)
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
