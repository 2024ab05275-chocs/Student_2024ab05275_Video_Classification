"""
data_loader.py
============================================================
This file serves TWO PURPOSES:

PART A: DATASET PREPARATION (UCF-101 SUBSET)
------------------------------------------------------------
- Cleans dataset directory
- Filters and validates videos
- Copies videos into flat class-wise folders
- Generates train / val / test splits
- Writes dataset metadata

PART B: CLASSICAL ML DATA LOADING
------------------------------------------------------------
- Reads prepared dataset
- Uses split files (train.txt / val.txt / test.txt)
- Extracts classical color features
- Returns NumPy arrays for ML training

IMPORTANT:
- This file is intentionally long but modular
- Dataset preparation runs ONLY when executed as a script
- Feature loading functions are used from notebooks
============================================================
"""

import shutil
import random
from pathlib import Path
import cv2
import numpy as np

# Import classical feature extractor
from feature_extraction import extract_video_color_features

# ============================================================
# CONFIGURATION SECTION
# ============================================================

# Resolve project root dynamically (one level above /code)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main dataset directory
DATASET_DIR = PROJECT_ROOT / "dataset"

# ------------------------------------------------------------
# SOURCE DATA (DO NOT MODIFY)
# ------------------------------------------------------------

SOURCE_RAR = DATASET_DIR / "UCF101.rar"
RAW_UCF_DIR = DATASET_DIR / "UCF-101"

# ------------------------------------------------------------
# CLASS DEFINITIONS
# ------------------------------------------------------------

CLASSES = {
    "class_1_Basketball": "Basketball",
    "class_2_Biking": "Biking",
    "class_3_WalkingWithDog": "WalkingWithDog"
}

# Create class-name → label mapping
CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(CLASSES.keys())}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# ------------------------------------------------------------
# DATASET SIZE CONSTRAINTS
# ------------------------------------------------------------

MIN_VIDEOS = 20
MAX_VIDEOS = 60

# ------------------------------------------------------------
# VIDEO QUALITY CONSTRAINTS
# ------------------------------------------------------------

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov"]
MIN_DURATION = 5
MAX_DURATION = 60
MIN_HEIGHT = 240

# ------------------------------------------------------------
# TRAIN / VAL / TEST SPLIT CONFIGURATION
# ------------------------------------------------------------

SPLIT_RATIO = (0.7, 0.15, 0.15)
SEED = 42
SPLITS_DIR = DATASET_DIR / "splits"

random.seed(SEED)

# ============================================================
# LOGGING UTILITY
# ============================================================

def log(msg):
    """Simple logging helper."""
    print(msg, flush=True)

# ============================================================
# DATASET PREPARATION FUNCTIONS (UNCHANGED)
# ============================================================

def clean_dataset():
    """Remove generated dataset files while preserving raw data."""
    log("🧹 Cleaning old dataset (preserving raw files)")

    for item in DATASET_DIR.iterdir():
        if item.name in ["UCF101.rar", "UCF-101"]:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def create_class_folders():
    """Create flat class-wise dataset folders."""
    for cls in CLASSES:
        (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)


def list_videos(folder: Path):
    """List all video files in a directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.glob(f"*{ext.lower()}"))
        videos.extend(folder.glob(f"*{ext.upper()}"))
    return videos


def is_video_valid(video: Path) -> bool:
    """Validate video by duration, resolution, and readability."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    if fps <= 0:
        return False

    duration = frames / fps
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return False
    if height < MIN_HEIGHT:
        return False

    return True


def copy_videos():
    """Copy validated videos from UCF-101 into flat folders."""
    log("📦 Copying valid videos")

    for target_cls, raw_cls in CLASSES.items():
        src_dir = RAW_UCF_DIR / raw_cls
        dst_dir = DATASET_DIR / target_cls

        if not src_dir.exists():
            log(f"⚠️ Missing raw folder: {raw_cls}")
            continue

        videos = list_videos(src_dir)
        random.shuffle(videos)

        valid = []
        for v in videos:
            if is_video_valid(v):
                valid.append(v)
            if len(valid) >= MAX_VIDEOS:
                break

        for v in valid:
            shutil.copy2(v, dst_dir / v.name)

        log(f"✔ {target_cls}: {len(valid)} videos copied")


def create_splits():
    """Create train/val/test split files."""
    train_f = open(SPLITS_DIR / "train.txt", "w")
    val_f = open(SPLITS_DIR / "val.txt", "w")
    test_f = open(SPLITS_DIR / "test.txt", "w")

    for cls in CLASSES:
        videos = list_videos(DATASET_DIR / cls)
        random.shuffle(videos)

        n = len(videos)
        n_train = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])

        for v in videos[:n_train]:
            train_f.write(f"{cls}/{v.name}\n")
        for v in videos[n_train:n_train + n_val]:
            val_f.write(f"{cls}/{v.name}\n")
        for v in videos[n_train + n_val:]:
            test_f.write(f"{cls}/{v.name}\n")

    train_f.close()
    val_f.close()
    test_f.close()

    log("📑 Dataset splits created")


def write_metadata():
    """Write dataset metadata files."""
    (DATASET_DIR / "dataset_info.txt").write_text(
        "UCF-101 Manual Subset\n"
        f"Classes: {', '.join(CLASSES.keys())}\n"
        "Duration: 5–60s | Resolution ≥240p\n"
    )

# ============================================================
# CLASSICAL ML DATA LOADING (NEW SECTION)
# ============================================================

def load_split(
    split_name: str,
    color_space: str = "HSV",
    bins: int = 16,
    max_frames: int = 50
):
    """
    Load a dataset split (train / val / test) and extract features.

    Args:
        split_name (str): One of {'train', 'val', 'test'}
        color_space (str): Color space for feature extraction
        bins (int): Histogram bins per channel
        max_frames (int): Frames per video

    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
    """
    split_file = SPLITS_DIR / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    X, y = [], []

    with open(split_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        rel_path = line.strip()
        class_name = rel_path.split("/")[0]
        video_path = DATASET_DIR / rel_path

        features = extract_video_color_features(
            video_path=str(video_path),
            color_space=color_space,
            bins=bins,
            max_frames=max_frames
        )

        X.append(features)
        y.append(CLASS_TO_LABEL[class_name])

    return np.vstack(X), np.array(y)


# ============================================================
# MAIN PIPELINE (DATASET PREPARATION ONLY)
# ============================================================

if __name__ == "__main__":
    log("🚀 Preparing UCF-101 dataset")

    if all((DATASET_DIR / cls).exists() for cls in CLASSES):
        log("✅ Dataset already prepared. Skipping.")
    else:
        clean_dataset()
        create_class_folders()
        copy_videos()
        create_splits()
        write_metadata()
        log("🎉 Dataset preparation complete")

    log("📌 Script finished")
