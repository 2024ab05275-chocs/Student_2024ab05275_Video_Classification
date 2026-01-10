"""
feature_extraction.py
-------------------------------------------------
This file contains classical (hand-crafted) feature
extraction functions for video-based activity recognition.

Implemented Features:
1. Color Histograms (RGB / HSV) per frame
2. Average Color Distribution across the video
3. Color Moments (Mean, Variance, Skewness)

The extracted features convert a variable-length video
into a fixed-length numerical feature vector, suitable
for classical machine learning models.
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

    Color moments are statistical descriptors that
    summarize the intensity distribution of pixels.

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
    Extract a normalized color histogram from a single frame.

    Args:
        frame (np.ndarray): Input frame in BGR format
        color_space (str): 'RGB' or 'HSV'
        bins (int): Number of bins per channel

    Returns:
        np.ndarray: Histogram feature vector
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
        np.ndarray: 9-dimensional feature vector
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

    Steps:
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
    Internally calls extract_video_color_features.
    """
    return extract_video_color_features(video_path, **kwargs)


# =================================================
# MAIN METHOD (SELF-TEST)
# =================================================

def main():
    """
    Simple sanity check for feature extraction.

    This method:
    - Finds one sample video from the dataset
    - Extracts features
    - Prints feature dimensions

    This does NOT train any model.
    """
    print("🔍 Running feature_extraction self-test")

    # Locate dataset directory dynamically
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"

    # Find first available video
    sample_video = None
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir() and class_dir.name.startswith("class_"):
            videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
            if videos:
                sample_video = videos[0]
                break

    if sample_video is None:
        print("❌ No sample video found for testing")
        return

    print(f"🎬 Sample video: {sample_video.name}")

    # Extract features
    features = extract_video_features(
        str(sample_video),
        color_space="HSV",
        bins=16,
        max_frames=30
    )

    print("✅ Feature extraction successful")
    print(f"📐 Feature vector length: {features.shape[0]}")
    print(f"📊 Feature vector (first 10 values):\n{features[:10]}")


if __name__ == "__main__":
    main()
