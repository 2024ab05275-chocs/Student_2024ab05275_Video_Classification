import shutil
import random
from pathlib import Path
import cv2

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

# SOURCE FILE (NEVER DELETE)
SOURCE_RAR = DATASET_DIR / "UCF101.rar"

# RAW MANUALLY EXTRACTED DATA
RAW_UCF_DIR = DATASET_DIR / "UCF-101"

# Class mapping: index + name
CLASSES = {
    "class_1_Basketball": "Basketball",
    "class_2_Biking": "Biking",
    "class_3_WalkingWithDog": "WalkingWithDog"
}

# Dataset constraints
MIN_VIDEOS = 20
MAX_VIDEOS = 60

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov"]
MIN_DURATION = 5       # seconds
MAX_DURATION = 60
MIN_HEIGHT = 240       # pixels

SPLIT_RATIO = (0.7, 0.15, 0.15)
SEED = 42

SPLITS_DIR = DATASET_DIR / "splits"
random.seed(SEED)

# =========================
# HELPERS
# =========================


def log(msg):
    print(msg, flush=True)


def clean_dataset():
    """
    Delete everything in dataset EXCEPT:
    - UCF101.rar
    - UCF-101 (raw extracted videos)
    """
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
    for cls in CLASSES:
        (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)


def list_videos(folder: Path):
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.glob(f"*{ext.lower()}"))
        videos.extend(folder.glob(f"*{ext.upper()}"))
    return videos


def is_video_valid(video: Path) -> bool:
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
    log("📦 Copying valid videos from UCF-101")

    for target_cls, raw_cls in CLASSES.items():
        src_dir = RAW_UCF_DIR / raw_cls
        dst_dir = DATASET_DIR / target_cls

        if not src_dir.exists():
            log(f"⚠️  WARNING: Missing raw class folder: {raw_cls}")
            continue

        videos = list_videos(src_dir)
        random.shuffle(videos)

        valid = []
        for v in videos:
            if is_video_valid(v):
                valid.append(v)
            if len(valid) >= MAX_VIDEOS:
                break

        if len(valid) < MIN_VIDEOS:
            log(f"⚠️  {raw_cls}: only {len(valid)} valid videos found")

        for v in valid:
            shutil.copy2(v, dst_dir / v.name)

        log(f"✔ {target_cls}: {len(valid)} videos copied")


def create_splits():
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

    log("📑 Train/Val/Test splits created")


def write_metadata():
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


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    log("🚀 Preparing UCF-101 subset (flat dataset structure)")
    clean_dataset()
    create_class_folders()
    copy_videos()
    create_splits()
    write_metadata()
    log("🎉 Dataset ready for classical & deep learning models")