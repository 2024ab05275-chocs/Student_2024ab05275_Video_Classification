"""
feature_extraction.py
-------------------------------------------------
Classical (hand-crafted) feature extraction for
video-based activity recognition.

Implemented Features:
1. Color Histograms (RGB / HSV) per frame
2. Average Color Distribution across the video
3. Color Moments (Mean, Variance, Skewness)

Key Design Principles:
- Single source of truth for feature computation
- No duplicated histogram logic
- Visualization uses already-extracted features
- Suitable for classical ML pipelines
"""

import cv2
import numpy as np
from typing import List
from pathlib import Path


# =================================================
# Helper Function: Color Moments
# =================================================

def _compute_color_moments(channel: np.ndarray) -> List[float]:
    """
    Compute color moments for a single image channel.

    Args:
        channel (np.ndarray): Single color channel

    Returns:
        List[float]: [mean, variance, skewness]
    """
    channel = channel.astype(np.float32)

    mean = np.mean(channel)
    variance = np.var(channel)
    std_dev = np.sqrt(variance) + 1e-6  # Prevent divide-by-zero
    skewness = np.mean(((channel - mean) / std_dev) ** 3)

    return [mean, variance, skewness]


# =================================================
# Frame-Level Feature Extraction
# =================================================

def extract_color_histogram(
    frame: np.ndarray,
    color_space: str = "HSV",
    bins: int = 16
) -> np.ndarray:
    """
    Extract normalized color histogram from a single frame.

    Args:
        frame (np.ndarray): Input frame in BGR format
        color_space (str): 'RGB' or 'HSV'
        bins (int): Number of bins per channel

    Returns:
        np.ndarray: Concatenated histogram feature vector
    """
    if color_space == "RGB":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif color_space == "HSV":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("color_space must be 'RGB' or 'HSV'")

    histogram_features = []

    for channel_index in range(3):
        hist = cv2.calcHist(
            [image],
            [channel_index],
            None,
            [bins],
            [0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        histogram_features.append(hist)

    return np.concatenate(histogram_features)


def extract_color_moments(
    frame: np.ndarray,
    color_space: str = "HSV"
) -> np.ndarray:
    """
    Extract color moments from a single frame.

    Args:
        frame (np.ndarray): Input frame in BGR format
        color_space (str): 'RGB' or 'HSV'

    Returns:
        np.ndarray: 9-dimensional color moment vector
    """
    if color_space == "RGB":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif color_space == "HSV":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("color_space must be 'RGB' or 'HSV'")

    moments = []

    for channel_index in range(3):
        moments.extend(
            _compute_color_moments(image[:, :, channel_index])
        )

    return np.array(moments)


# =================================================
# Video-Level Feature Extraction
# =================================================

def extract_video_color_features(
    video_path: str,
    color_space: str = "HSV",
    bins: int = 16,
    max_frames: int = None
) -> np.ndarray:
    """
    Extract color-based features from an entire video.

    Pipeline:
    1. Read video frame-by-frame
    2. Extract histogram + color moments per frame
    3. Average features across all frames

    Args:
        video_path (str): Path to video file
        color_space (str): 'RGB' or 'HSV'
        bins (int): Histogram bins per channel
        max_frames (int): Optional frame limit

    Returns:
        np.ndarray: Fixed-length video feature vector
    """
    cap = cv2.VideoCapture(video_path)

    histogram_sum = None
    moment_sum = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hist = extract_color_histogram(frame, color_space, bins)
        moments = extract_color_moments(frame, color_space)

        if histogram_sum is None:
            histogram_sum = np.zeros_like(hist)
            moment_sum = np.zeros_like(moments)

        histogram_sum += hist
        moment_sum += moments
        frame_count += 1

        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    avg_histogram = histogram_sum / frame_count
    avg_moments = moment_sum / frame_count

    return np.concatenate([avg_histogram, avg_moments])


# =================================================
# PUBLIC API WRAPPER
# =================================================

def extract_video_features(video_path, **kwargs):
    """
    Wrapper function used by classical ML pipelines.
    """
    return extract_video_color_features(video_path, **kwargs)


# =================================================
# Visualization (NO DUPLICATE LOGIC)
# =================================================

def plot_color_histogram_from_features(
    hist_features: np.ndarray,
    bins: int = 16,
    color_space: str = "HSV"
):
    """
    Plot color histograms using already extracted
    histogram features.

    This function does NOT recompute histograms.

    Args:
        hist_features (np.ndarray): Histogram feature vector
        bins (int): Number of bins per channel
        color_space (str): 'RGB' or 'HSV'
    """
    import matplotlib.pyplot as plt

    channel_names = ["R", "G", "B"] if color_space == "RGB" else ["H", "S", "V"]

    plt.figure(figsize=(8, 4))

    for i, ch in enumerate(channel_names):
        start = i * bins
        end = (i + 1) * bins
        plt.plot(hist_features[start:end], label=ch)

    plt.title(f"{color_space} Color Histogram (Single Frame)")
    plt.xlabel("Bin Index")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =================================================
# MAIN METHOD (SELF-TEST + VISUALIZATION)
# =================================================

def main():
    """
    Standalone sanity test for feature extraction.

    - Loads one sample video
    - Plots color histogram of first frame
    - Extracts full video features
    """
    print("🔍 Running feature_extraction self-test")

    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"

    sample_video = None
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir() and class_dir.name.startswith("class_"):
            videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
            if videos:
                sample_video = videos[0]
                break

    if sample_video is None:
        print("❌ No sample video found")
        return

    print(f"🎬 Sample video: {sample_video.name}")

    cap = cv2.VideoCapture(str(sample_video))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read frame")
        return

    # Extract histogram features (single source of truth)
    hist_features = extract_color_histogram(
        frame,
        color_space="HSV",
        bins=16
    )

    # Plot histogram (no recomputation)
    plot_color_histogram_from_features(
        hist_features,
        bins=16,
        color_space="HSV"
    )

    # Full video feature extraction
    features = extract_video_features(
        str(sample_video),
        color_space="HSV",
        bins=16,
        max_frames=30
    )

    print("✅ Feature extraction successful")
    print(f"📐 Feature vector length: {features.shape[0]}")
    print(f"📊 First 10 feature values:\n{features[:10]}")


if __name__ == "__main__":
    main()
