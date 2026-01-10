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

Key Design Principles
-------------------------------------------------
- Single-pass video processing
- Fixed-length output representation
- No duplicate computation
- Interpretable features
- Robust to varying video length
"""

# =================================================
# IMPORTS
# =================================================

import cv2                          # Video I/O and image processing
import numpy as np                  # Numerical computation
from pathlib import Path            # Filesystem handling
from typing import List             # Type hints

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

# =================================================
# LOGGING UTILITY
# =================================================


def log(message: str) -> None:
    """
    Centralized lightweight logger.
    Keeps output consistent and readable.
    """
    print(f"[INFO] {message}", flush=True)


# =================================================
# COLOR FEATURE HELPERS
# =================================================

def _compute_color_moments(channel: np.ndarray) -> List[float]:
    """
    Compute statistical color moments for a single channel.

    Moments describe the distribution of pixel intensities.

    Returns:
        [mean, variance, skewness]
    """
    channel = channel.astype(np.float32)

    mean = np.mean(channel)
    variance = np.var(channel)
    std_dev = np.sqrt(variance) + 1e-6
    skewness = np.mean(((channel - mean) / std_dev) ** 3)

    return [mean, variance, skewness]


# =================================================
# FRAME-LEVEL COLOR FEATURES
# =================================================

def extract_color_histogram(
    frame: np.ndarray,
    color_space: str = "HSV",
    bins: int = 16
) -> np.ndarray:
    """
    Extract normalized color histogram from a frame.

    Each channel contributes `bins` values.
    """
    if color_space == "RGB":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    features = []

    for channel_idx in range(3):
        hist = cv2.calcHist(
            [image],
            [channel_idx],
            None,
            [bins],
            [0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    return np.concatenate(features)


def extract_color_moments(
    frame: np.ndarray,
    color_space: str = "HSV"
) -> np.ndarray:
    """
    Extract color moments (mean, variance, skewness)
    from each color channel.
    """
    if color_space == "RGB":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    moments = []
    for ch in range(3):
        moments.extend(_compute_color_moments(image[:, :, ch]))

    return np.array(moments)


# =================================================
# TEXTURE FEATURES (FRAME-LEVEL)
# =================================================

def extract_glcm_features(gray_frame: np.ndarray) -> np.ndarray:
    """
    Extract GLCM-based texture descriptors.

    Measures spatial intensity relationships.
    """
    glcm = graycomatrix(
        gray_frame,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    properties = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    )

    return np.array([graycoprops(glcm, p)[0, 0] for p in properties])


def extract_lbp_features(
    gray_frame: np.ndarray,
    radius: int = 1,
    points: int = 8
) -> np.ndarray:
    """
    Extract Local Binary Pattern histogram.

    Uniform LBP ensures rotation invariance.
    """
    lbp = local_binary_pattern(
        gray_frame,
        points,
        radius,
        method="uniform"
    )

    n_bins = points + 2
    lbp = np.clip(lbp.astype(np.int32), 0, n_bins - 1)

    hist = np.bincount(lbp.ravel(), minlength=n_bins).astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist


def extract_gabor_features(gray_frame: np.ndarray) -> np.ndarray:
    """
    Extract Gabor filter statistics (mean, variance).
    """
    features = []

    for theta in (0, np.pi / 4):
        for sigma in (1.0, 3.0):
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=sigma,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0
            )
            filtered = cv2.filter2D(gray_frame, cv2.CV_32F, kernel)
            features.extend([filtered.mean(), filtered.var()])

    return np.array(features)


# =================================================
# MOTION FEATURES
# =================================================

def extract_motion_statistics(diff_frame: np.ndarray) -> np.ndarray:
    """
    Compute statistical motion descriptors.
    """
    diff_frame = diff_frame.astype(np.float32)

    return np.array([
        np.mean(diff_frame),
        np.var(diff_frame),
        np.max(diff_frame)
    ])


def extract_motion_histogram(
    diff_frame: np.ndarray,
    bins: int = 16
) -> np.ndarray:
    """
    Histogram of motion intensity.
    """
    hist, _ = np.histogram(diff_frame, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist


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
    Extract all classical features from a video.

    Features are averaged across frames to ensure
    fixed-length output.
    """
    log(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    color_hist_sum = color_moment_sum = None
    glcm_sum = lbp_sum = gabor_sum = None
    motion_stat_sum = motion_hist_sum = None

    prev_gray = None
    frame_count = motion_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ch = extract_color_histogram(frame, color_space, bins)
        cm = extract_color_moments(frame, color_space)
        glcm = extract_glcm_features(gray)
        lbp = extract_lbp_features(gray)
        gabor = extract_gabor_features(gray)

        if color_hist_sum is None:
            log("Initializing feature accumulators")
            color_hist_sum = np.zeros_like(ch)
            color_moment_sum = np.zeros_like(cm)
            glcm_sum = np.zeros_like(glcm)
            lbp_sum = np.zeros_like(lbp)
            gabor_sum = np.zeros_like(gabor)

        color_hist_sum += ch
        color_moment_sum += cm
        glcm_sum += glcm
        lbp_sum += lbp
        gabor_sum += gabor
        frame_count += 1

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            m_stat = extract_motion_statistics(diff)
            m_hist = extract_motion_histogram(diff)

            if motion_stat_sum is None:
                motion_stat_sum = np.zeros_like(m_stat)
                motion_hist_sum = np.zeros_like(m_hist)

            motion_stat_sum += m_stat
            motion_hist_sum += m_hist
            motion_count += 1

        prev_gray = gray

        if max_frames and frame_count >= max_frames:
            log(f"Frame limit reached: {max_frames}")
            break

    cap.release()

    if frame_count == 0:
        raise RuntimeError("No frames were read from video")

    log(f"Frames processed: {frame_count}")
    log(f"Motion frames processed: {motion_count}")

    features = np.concatenate([
        color_hist_sum / frame_count,
        color_moment_sum / frame_count,
        glcm_sum / frame_count,
        lbp_sum / frame_count,
        gabor_sum / frame_count,
        motion_stat_sum / max(motion_count, 1),
        motion_hist_sum / max(motion_count, 1)
    ])

    log(f"Final feature vector length: {features.shape[0]}")
    return features


# =================================================
# MAIN: SELF-TEST
# =================================================

def main() -> None:
    """
    Standalone verification run.
    """
    log("Running feature extraction self-test")

    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset"

    sample_video = None
    for cls_dir in dataset.iterdir():
        if cls_dir.is_dir():
            videos = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi"))
            if videos:
                sample_video = videos[0]
                break

    if sample_video is None:
        log("No sample video found in dataset")
        return

    extract_video_features(str(sample_video), max_frames=30)
    log("Self-test completed successfully")


if __name__ == "__main__":
    main()
