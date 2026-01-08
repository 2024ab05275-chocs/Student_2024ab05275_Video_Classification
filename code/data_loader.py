import shutil
import random
from pathlib import Path
import cv2

# ============================================================
# CONFIGURATION SECTION
# ============================================================
# This section defines all configurable parameters for the
# dataset preparation pipeline. Modifying values here allows
# easy reuse of the script for other datasets or experiments.
# ============================================================

# Resolve project root dynamically (two levels above this file)
# This ensures the script works regardless of where it is executed from
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main dataset directory
DATASET_DIR = PROJECT_ROOT / "dataset"

# ------------------------------------------------------------
# SOURCE DATA (DO NOT MODIFY / DELETE)
# ------------------------------------------------------------

# Original compressed dataset archive
# This file is preserved for reproducibility and backup
SOURCE_RAR = DATASET_DIR / "UCF101.rar"

# Raw manually extracted UCF-101 dataset directory
# Contains original class-wise folders from UCF-101
RAW_UCF_DIR = DATASET_DIR / "UCF-101"

# ------------------------------------------------------------
# CLASS DEFINITIONS
# ------------------------------------------------------------
# Mapping of target class folder names to original UCF-101
# class names. This allows:
# - Controlled class selection
# - Easy relabeling
# - Maximum of 5 classes (as per assignment requirement)
# ------------------------------------------------------------

CLASSES = {
    "class_1_Basketball": "Basketball",
    "class_2_Biking": "Biking",
    "class_3_WalkingWithDog": "WalkingWithDog"
}

# ------------------------------------------------------------
# DATASET SIZE CONSTRAINTS
# ------------------------------------------------------------
# These values ensure compliance with dataset requirements:
# - Minimum videos per class for classical ML
# - Upper bound to keep dataset size manageable
# ------------------------------------------------------------

MIN_VIDEOS = 20     # Minimum acceptable videos per class
MAX_VIDEOS = 60     # Maximum videos copied per class

# ------------------------------------------------------------
# VIDEO QUALITY CONSTRAINTS
# ------------------------------------------------------------
# Used to filter out low-quality or unsuitable videos
# ------------------------------------------------------------

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov"]  # Allowed formats
MIN_DURATION = 5       # Minimum video length in seconds
MAX_DURATION = 60      # Maximum video length in seconds
MIN_HEIGHT = 240       # Minimum vertical resolution (240p)

# ------------------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT CONFIGURATION
# ------------------------------------------------------------

SPLIT_RATIO = (0.7, 0.15, 0.15)  # Train / Val / Test split
SEED = 42                       # Fixed seed for reproducibility

# Directory to store split files
SPLITS_DIR = DATASET_DIR / "splits"

# Set random seed so dataset splits are deterministic
random.seed(SEED)

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def log(msg):
    """
    Simple logging helper to print messages immediately.
    flush=True ensures logs appear in real-time during execution.
    """
    print(msg, flush=True)


def clean_dataset():
    """
    Cleans the dataset directory by removing all generated data
    while preserving:
    - Original dataset archive (UCF101.rar)
    - Raw extracted dataset (UCF-101)

    This ensures idempotent runs of the script.
    """
    log("🧹 Cleaning old dataset (preserving raw files)")

    for item in DATASET_DIR.iterdir():
        # Skip original data sources
        if item.name in ["UCF101.rar", "UCF-101"]:
            continue

        # Remove generated folders or files
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Recreate splits directory after cleanup
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def create_class_folders():
    """
    Creates flat directory structure for each target class.
    Each folder will store validated video files.
    """
    for cls in CLASSES:
        (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)


def list_videos(folder: Path):
    """
    Lists all video files in a folder matching allowed extensions.
    Handles both lowercase and uppercase extensions to avoid misses.

    Args:
        folder (Path): Directory to search for videos

    Returns:
        List[Path]: List of video file paths
    """
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.glob(f"*{ext.lower()}"))
        videos.extend(folder.glob(f"*{ext.upper()}"))
    return videos


def is_video_valid(video: Path) -> bool:
    """
    Validates a video file based on:
    - Openability
    - FPS validity
    - Duration (5–60 seconds)
    - Resolution (≥240p)

    Args:
        video (Path): Path to video file

    Returns:
        bool: True if video meets all constraints
    """
    cap = cv2.VideoCapture(str(video))

    # Check if video can be opened
    if not cap.isOpened():
        return False

    # Extract video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.release()

    # Invalid FPS indicates corrupted or unsupported video
    if fps <= 0:
        return False

    # Compute duration in seconds
    duration = frames / fps

    # Apply duration and resolution constraints
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return False
    if height < MIN_HEIGHT:
        return False

    return True


def copy_videos():
    """
    Copies validated videos from raw UCF-101 directories
    into flat class-specific folders.

    - Randomly shuffles videos
    - Applies validation checks
    - Enforces min/max videos per class
    """
    log("📦 Copying valid videos from UCF-101")

    for target_cls, raw_cls in CLASSES.items():
        src_dir = RAW_UCF_DIR / raw_cls
        dst_dir = DATASET_DIR / target_cls

        # Warn if raw class folder is missing
        if not src_dir.exists():
            log(f"⚠️  WARNING: Missing raw class folder: {raw_cls}")
            continue

        videos = list_videos(src_dir)
        random.shuffle(videos)

        valid = []
        for v in videos:
            if is_video_valid(v):
                valid.append(v)

            # Stop once maximum required videos are collected
            if len(valid) >= MAX_VIDEOS:
                break

        # Warn if minimum dataset size is not met
        if len(valid) < MIN_VIDEOS:
            log(f"⚠️  {raw_cls}: only {len(valid)} valid videos found")

        # Copy validated videos to target directory
        for v in valid:
            shutil.copy2(v, dst_dir / v.name)

        log(f"✔ {target_cls}: {len(valid)} videos copied")


def create_splits():
    """
    Generates train, validation, and test splits.
    Each split file contains relative paths to videos.

    Split is done class-wise to preserve class balance.
    """
    train_f = open(SPLITS_DIR / "train.txt", "w")
    val_f = open(SPLITS_DIR / "val.txt", "w")
    test_f = open(SPLITS_DIR / "test.txt", "w")

    for cls in CLASSES:
        videos = list_videos(DATASET_DIR / cls)
        random.shuffle(videos)

        n = len(videos)
        n_train = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])

        # Write train split
        for v in videos[:n_train]:
            train_f.write(f"{cls}/{v.name}\n")

        # Write validation split
        for v in videos[n_train:n_train + n_val]:
            val_f.write(f"{cls}/{v.name}\n")

        # Remaining videos go to test split
        for v in videos[n_train + n_val:]:
            test_f.write(f"{cls}/{v.name}\n")

    train_f.close()
    val_f.close()
    test_f.close()

    log("📑 Train/Val/Test splits created")


def dataset_already_prepared() -> bool:
    """
    Checks whether dataset class folders already exist
    and contain video files.

    Returns:
        bool: True if all class folders exist and are non-empty
    """
    for cls in CLASSES:
        class_dir = DATASET_DIR / cls

        # Folder must exist
        if not class_dir.exists():
            return False

        # Folder must contain at least one video file
        videos = list_videos(class_dir)
        if len(videos) == 0:
            return False

    return True


def write_metadata():
    """
    Writes dataset documentation files describing:
    - Dataset source
    - Class labels
    - Video constraints
    - Classification type
    """
    (DATASET_DIR / "dataset_info.txt").write_text(
        "UCF-101 Manual Subset\n"
        f"Classes: {', '.join(CLASSES.keys())}\n"
        "Video formats: MP4 / AVI / MOV\n"
        "Duration: 5–60 seconds\n"
        "Resolution: ≥240p\n"
        "Multi-class classification (≤5 classes)\n"
    )

    (DATASET_DIR / "README.md").write_text(
        "# Video Classification Dataset\n\n"
        "Prepared from UCF-101 (manual extraction).\n"
        "Validated for duration, resolution, and format.\n"
        "Train/Val/Test splits automatically generated.\n"
    )


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================

if __name__ == "__main__":
    log("🚀 Preparing UCF-101 subset (flat dataset structure)")

    # --------------------------------------------------------
    # SAFETY CHECK:
    # Skip processing if dataset already exists and is valid
    # --------------------------------------------------------
    if dataset_already_prepared():
        log("✅ Dataset already prepared. Skipping data processing steps.")
    else:
        # Step 1: Clean previously generated data
        clean_dataset()

        # Step 2: Create class-wise output folders
        create_class_folders()

        # Step 3: Validate and copy videos
        copy_videos()

        # Step 4: Generate train/val/test splits
        create_splits()

        # Step 5: Write dataset metadata and documentation
        write_metadata()

        log("🎉 Dataset prepared successfully")

    log("📌 Script execution completed")