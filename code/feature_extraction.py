"""
feature_extraction.py
-------------------------------------------------
Classical (hand-crafted) feature extraction for
video-based activity recognition.

This module implements traditional computer vision
features that transform a variable-length video
sequence into a fixed-length numerical feature vector,
making it suitable for classical machine learning
models such as SVM, Random Forest, and k-NN.

Implemented Feature Categories
-------------------------------------------------
1. Color Features
   - RGB / HSV color histograms
   - Color moments (mean, variance, skewness)

2. Texture Features
   - Gray Level Co-occurrence Matrix (GLCM)
   - Local Binary Patterns (LBP)
   - Gabor filter responses

3. Motion Features
   - Frame differencing
   - Motion statistics
   - Motion intensity histogram

4. Temporal Features
   - Statistical measures over feature sequences
   - Frame-to-frame variation analysis
   - Temporal gradients and patterns

Key Design Principles
-------------------------------------------------
- Single-pass video processing
- No duplicate computation
- Fixed-length feature vector
- Robust to variable video length
- Interpretable, classical features
"""

# =================================================
# IMPORTS
# =================================================

import cv2
import numpy as np
from pathlib import Path
from typing import List

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

# =================================================
# LOGGING UTILITY
# =================================================


def log(message: str) -> None:
    """Centralized logging helper."""
    print(f"[INFO] {message}", flush=True)


# =================================================
# COLOR FEATURE HELPERS
# =================================================

def _compute_color_moments(channel: np.ndarray) -> List[float]:
    """
    Compute mean, variance and skewness of a channel.
    """
    channel = channel.astype(np.float32)

    mean = np.mean(channel)
    variance = np.var(channel)
    std = np.sqrt(variance) + 1e-6
    skewness = np.mean(((channel - mean) / std) ** 3)

    return [mean, variance, skewness]


# =================================================
# FRAME-LEVEL COLOR FEATURES
# =================================================

def extract_color_histogram(
    frame: np.ndarray,
    color_space: str = "HSV",
    bins: int = 16
) -> np.ndarray:
    """Extract normalized color histogram per frame."""
    image = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if color_space == "RGB"
        else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    )

    features = []
    for ch in range(3):
        hist = cv2.calcHist([image], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    return np.concatenate(features)


def extract_color_moments(
    frame: np.ndarray,
    color_space: str = "HSV"
) -> np.ndarray:
    """Extract color moments per frame."""
    image = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if color_space == "RGB"
        else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    )

    moments = []
    for ch in range(3):
        moments.extend(_compute_color_moments(image[:, :, ch]))

    return np.array(moments)


# =================================================
# TEXTURE FEATURES
# =================================================

def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    """Extract GLCM texture descriptors."""
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    props = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    )

    return np.array([graycoprops(glcm, p)[0, 0] for p in props])


def extract_lbp_features(
    gray: np.ndarray,
    radius: int = 1,
    points: int = 8
) -> np.ndarray:
    """Extract uniform LBP histogram."""
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    n_bins = points + 2

    lbp = np.clip(lbp.astype(np.int32), 0, n_bins - 1)
    hist = np.bincount(lbp.ravel(), minlength=n_bins).astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist


def extract_gabor_features(gray: np.ndarray) -> np.ndarray:
    """Extract Gabor filter mean & variance."""
    features = []

    for theta in (0, np.pi / 4):
        for sigma in (1.0, 3.0):
            kernel = cv2.getGaborKernel(
                (21, 21), sigma, theta, 10.0, 0.5, 0
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.extend([filtered.mean(), filtered.var()])

    return np.array(features)


# =================================================
# MOTION FEATURES
# =================================================

def extract_motion_statistics(diff: np.ndarray) -> np.ndarray:
    """Mean, variance and max of motion intensity."""
    diff = diff.astype(np.float32)
    return np.array([diff.mean(), diff.var(), diff.max()])


def extract_motion_histogram(diff: np.ndarray, bins: int = 16) -> np.ndarray:
    """Histogram of motion magnitude."""
    hist, _ = np.histogram(diff, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


# =================================================
# TEMPORAL FEATURES
# =================================================

def extract_temporal_statistics(sequence: np.ndarray) -> np.ndarray:
    """
    Compute temporal statistics over a feature sequence.

    Captures:
    - Mean
    - Standard deviation
    - Minimum
    - Maximum
    """
    return np.concatenate([
        sequence.mean(axis=0),
        sequence.std(axis=0),
        sequence.min(axis=0),
        sequence.max(axis=0),
    ])


def extract_temporal_gradients(sequence: np.ndarray) -> np.ndarray:
    """
    Analyze frame-to-frame variation using first-order
    temporal gradients.
    """
    gradients = np.diff(sequence, axis=0)
    return np.array([
        gradients.mean(),
        gradients.std(),
        gradients.max()
    ])


# =================================================
# VIDEO-LEVEL FEATURE AGGREGATION
# =================================================

def extract_video_features(
    video_path: str,
    color_space: str = "HSV",
    bins: int = 16,
    max_frames: int | None = None
) -> np.ndarray:
    """
    Extract all classical features including temporal
    statistics from a video.
    """
    rel_path = Path(video_path)
    if "dataset" in rel_path.parts:
        rel_path = Path(*rel_path.parts[rel_path.parts.index("dataset") + 1:])

    log(f"* processing Video file : {rel_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_features = []
    motion_features = []

    prev_gray = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Per-frame spatial features
        spatial = np.concatenate([
            extract_color_histogram(frame, color_space, bins),
            extract_color_moments(frame, color_space),
            extract_glcm_features(gray),
            extract_lbp_features(gray),
            extract_gabor_features(gray)
        ])

        frame_features.append(spatial)

        # Motion features
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = np.concatenate([
                extract_motion_statistics(diff),
                extract_motion_histogram(diff)
            ])
            motion_features.append(motion)

        prev_gray = gray
        frame_count += 1

        if max_frames and frame_count >= max_frames:
            #  log(f"Frame limit reached: {max_frames}")
            break

    cap.release()

    if frame_count < 2:
        raise RuntimeError("Insufficient frames for temporal analysis")

    #  log(f"Frames processed: {frame_count}")

    frame_features = np.vstack(frame_features)
    motion_features = np.vstack(motion_features)

    # Temporal aggregation
    temporal_stats = extract_temporal_statistics(frame_features)
    temporal_grad = extract_temporal_gradients(frame_features)

    motion_stats = extract_temporal_statistics(motion_features)

    final_features = np.concatenate([
        temporal_stats,
        temporal_grad,
        motion_stats
    ])

    #  log(f"Final feature vector length: {final_features.shape[0]}")
    return final_features


# =================================================
# MAIN: SELF TEST
# =================================================

def main() -> None:
    """Run sanity check on dataset."""
    log("Running feature extraction self-test")

    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset"

    for cls in dataset.iterdir():
        if cls.is_dir():
            videos = list(cls.glob("*.mp4")) + list(cls.glob("*.avi"))
            if videos:
                extract_video_features(str(videos[0]), max_frames=30)
                log("Self-test completed successfully")
                return

    log("No video found for self-test")


if __name__ == "__main__":
    main()
