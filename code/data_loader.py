"""
data_loader.py
============================================================
This file serves TWO PURPOSES:

PART A: DATASET PREPARATION (UCF-101 SUBSET)
------------------------------------------------------------
- Cleans dataset directory
- Filters and validates videos
- Copies videos into flat class-wise folders
- Restricts dataset size for classical ML
- Generates train / val / test splits
- Writes dataset metadata

PART B: CLASSICAL ML DATA LOADING + PREPROCESSING
------------------------------------------------------------
- Reads prepared dataset
- Uses split files (train.txt / val.txt / test.txt)
- Applies frame-level preprocessing
- Enforces temporal consistency
- Extracts classical color features
- Returns NumPy arrays for ML training

ADDITIONAL VERIFICATION STEP
------------------------------------------------------------
- Loads ALL videos from ALL classes after preparation
- Validates preprocessing and temporal alignment
- Logs final tensor shapes and label mapping

IMPORTANT:
- This file is intentionally long but modular
- Dataset preparation runs ONLY when executed as a script
- Data loading utilities are used when imported
- ALL preprocessing must happen inside this file
============================================================
"""

# ============================================================
# STANDARD LIBRARIES
# ============================================================

import math
import shutil
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# ============================================================
# THIRD-PARTY LIBRARIES
# ============================================================

import cv2
import numpy as np

# ============================================================
# PROJECT IMPORTS
# ============================================================

from feature_extraction import extract_video_features

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

RAW_UCF_DIR = DATASET_DIR / "UCF-101"
SOURCE_RAR = DATASET_DIR / "UCF101.rar"

# ------------------------------------------------------------
# CLASS DEFINITIONS
# ------------------------------------------------------------

CLASSES = {
    "class_1_Basketball": "Basketball",
    "class_2_Biking": "Biking",
    "class_3_WalkingWithDog": "WalkingWithDog",
}

CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(CLASSES)}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# ------------------------------------------------------------
# DATASET SIZE CONSTRAINTS
# ------------------------------------------------------------

MIN_VIDEOS = 20
MAX_VIDEOS = 60  # balanced classical ML dataset

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
np.random.seed(SEED)

# ============================================================
# LOGGING UTILITY
# ============================================================


def log(msg):
    """Simple logging helper."""
    print(msg, flush=True)

# ============================================================
# DATASET PREPARATION FUNCTIONS (UNCHANGED)
# ============================================================


def clean_dataset() -> None:
    """Remove generated dataset while preserving raw data."""
    log("🧹 Cleaning dataset directory")

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


def list_videos(folder: Path) -> List[Path]:
    """List all supported video files."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.glob(f"*{ext.lower()}"))
        videos.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(videos)


def is_video_valid(video: Path) -> bool:
    """Validate video by readability, duration, and resolution."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    if fps <= 0 or frames <= 0:
        return False

    duration = frames / fps
    if not (MIN_DURATION <= duration <= MAX_DURATION):
        return False

    if height < MIN_HEIGHT:
        return False

    return True


def copy_videos() -> None:
    """Copy validated videos into flat class folders."""
    log("📦 Copying validated videos")

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


def create_splits() -> None:
    """Create train / val / test split files."""
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


def write_metadata() -> None:
    """Write dataset metadata."""
    (DATASET_DIR / "dataset_info.txt").write_text(
        "UCF-101 Classical ML Subset\n"
        f"Classes: {', '.join(CLASSES.keys())}\n"
        f"Videos per class: {MAX_VIDEOS}\n"
        "Duration: 5–60s | Resolution ≥240p\n"
    )


# ============================================================
# PART B: CLASSICAL ML DATA LOADING
# ============================================================

def preprocess_frame(
    frame: np.ndarray,
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize, grayscale, normalize frame."""
    resized = cv2.resize(frame, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    return normalized[..., np.newaxis]


def load_split_data_and_extract_features(
    split_name: str,
    color_space: str = "HSV",
    bins: int = 16,
    max_frames: int = 50,
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
        for line in f:
            rel_path = line.strip()
            cls = rel_path.split("/")[0]
            video_path = DATASET_DIR / rel_path

            features = extract_video_features(
                video_path=str(video_path),
                color_space=color_space,
                bins=bins,
                max_frames=max_frames,
            )

            X.append(features)
            y.append(CLASS_TO_LABEL[cls])

    return np.vstack(X), np.array(y)


# ============================================================
# DATASET VERIFICATION LOADER (FULL VIDEOS)
# ============================================================

DEFAULT_FRAME_SIZE = (224, 224)
DEFAULT_MAX_FRAMES = 50


class VideoDataLoader:
    """
    Loads full video tensors for verification.
    """

    def __init__(self, dataset_root, frame_size, max_frames):
        self.dataset_root = Path(dataset_root)
        self.frame_size = frame_size
        self.max_frames = max_frames

    def _load_video(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        if not cap.isOpened():
            return frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def _fix_length(self, frames):
        if not frames:
            return []
        if len(frames) >= self.max_frames:
            idx = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            return [frames[i] for i in idx]
        return frames + [frames[-1]] * (self.max_frames - len(frames))

    def load_dataset(self):
        X, y = [], []

        class_dirs = sorted(
            d for d in self.dataset_root.iterdir()
            if d.is_dir() and d.name.startswith("class_")
        )

        class_to_label = {d.name: i for i, d in enumerate(class_dirs)}
        label_map = {v: k for k, v in class_to_label.items()}

        for cls_dir in class_dirs:
            label = class_to_label[cls_dir.name]
            for video in list_videos(cls_dir):
                frames = self._load_video(video)
                frames = self._fix_length(frames)
                if not frames:
                    continue

                processed = [
                    preprocess_frame(f, self.frame_size)
                    for f in frames
                ]

                X.append(np.stack(processed))
                y.append(label)

        return np.array(X), np.array(y), label_map


def compute_dataset_statistics(
    dataset_root: str,
    frame_sampling_rate: int = 1,
    split_ratio: tuple = (0.7, 0.15, 0.15)
):
    """
    Compute dataset statistics directly from dataset directory.

    Args:
        dataset_root (str): Root directory of dataset
        frame_sampling_rate (int): Extract 1 frame every N frames
        split_ratio (tuple): (train, val, test) ratios

    Returns:
        dict: Dataset statistics
    """
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"

    dataset_root = PROJECT_ROOT / "dataset"
    class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]

    num_classes = len(class_dirs)
    total_videos = 0
    total_frames = 0

    for class_dir in class_dirs:
        video_files = (
            list(class_dir.glob("*.avi")) +
            list(class_dir.glob("*.mp4")) +
            list(class_dir.glob("*.mov"))
        )

        total_videos += len(video_files)

        for video in video_files:
            cap = cv2.VideoCapture(str(video))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            total_frames += frame_count // frame_sampling_rate

    train_ratio, val_ratio, test_ratio = split_ratio

    train_videos = math.floor(total_videos * train_ratio)
    val_videos = math.floor(total_videos * val_ratio)
    test_videos = total_videos - train_videos - val_videos

    return {
        "Number of classes": num_classes,
        "Total videos": total_videos,
        "Total frames extracted": total_frames,
        "Train videos": train_videos,
        "Validation videos": val_videos,
        "Test videos": test_videos,
        "Split ratio": f"{int(train_ratio*100)}/"
                       f"{int(val_ratio*100)}/"
                       f"{int(test_ratio*100)}"
    }


def create_dataset_info(
    root_dir: str,
    dataset_url: str,
    description: str,
    stats: dict
):
    """
    Create dataset_info folder structure and populate metadata files.

    Args:
        root_dir (str): Project root directory
        dataset_url (str): URL or source of dataset
        description (str): Markdown description of dataset
        stats (dict): Dataset statistics (classes, videos, frames, etc.)
    """
    
    info_dir = PROJECT_ROOT / "dataset_info"
    print(info_dir)
    samples_dir = info_dir / "sample_frames"

    # Create directories
    info_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # dataset_url.txt
    (info_dir / "dataset_url.txt").write_text(
        f"{dataset_url}\n\nGenerated on: {datetime.now()}\n",
        encoding="utf-8"
    )

    # dataset_description.md
    (info_dir / "dataset_description.md").write_text(
        f"# Dataset Description\n\n{description}\n",
        encoding="utf-8"
    )

    # data_statistics.txt
    stats_text = ["Dataset Statistics\n", "-" * 20]
    for k, v in stats.items():
        stats_text.append(f"{k}: {v}")

    (info_dir / "data_statistics.txt").write_text(
        "\n".join(stats_text),
        encoding="utf-8"
    )

    print(f"[INFO] dataset_info created at: {info_dir}")

stats = compute_dataset_statistics(
    dataset_root= "dataset",
    frame_sampling_rate=5,
    split_ratio=SPLIT_RATIO
)

# ============================================================
# SCRIPT ENTRY POINT (PART A + VERIFICATION)
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

    # Create Dataset Information
    create_dataset_info(
            root_dir=".",
            dataset_url="https://www.crcv.ucf.edu/data/UCF101.php",
            description="""
            Video action recognition dataset.
            Frames sampled every 5 frames.
            Split into Train / Validation / Test sets.
            """,
            stats=stats
        )

    # --------------------------------------------------------
    # VERIFICATION STEP
    # --------------------------------------------------------

    log("🔍 Verifying prepared dataset")

    loader = VideoDataLoader(
        dataset_root=str(DATASET_DIR),
        frame_size=DEFAULT_FRAME_SIZE,
        max_frames=DEFAULT_MAX_FRAMES,
    )

    X, y, label_map = loader.load_dataset()

    log(f"📊 Final data shape X: {X.shape}")
    log(f"📊 Final label shape y: {y.shape}")
    log(f"🏷 Label map: {label_map}")

    log("📌 data_loader.py execution finished")
