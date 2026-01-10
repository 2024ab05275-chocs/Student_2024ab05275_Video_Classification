"""
feature_extraction.py
-------------------------------------------------
Classical (hand-crafted) feature extraction for
video-based activity recognition.

Implemented Feature Categories
-------------------------------------------------
1. Color Features
   - RGB / HSV color histograms per frame
   - Average color histogram across video
   - Color moments (mean, variance, skewness)

2. Texture Features
   - Gray Level Co-occurrence Matrix (GLCM)
   - Local Binary Patterns (LBP)
   - Gabor filter responses

All features convert a variable-length video into
a fixed-length numerical vector suitable for
classical machine learning models.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# =================================================
# Helper: Color Moments
# =================================================

def _compute_color_moments(channel: np.ndarray) -> List[float]:
    """
    Compute color moments for a single channel.

    Returns:
        [mean, variance, skewness]
    """
    channel = channel.astype(np.float32)

    mean = np.mean(channel)
    variance = np.var(channel)
    std = np.sqrt(variance) + 1e-6
    skewness = np.mean(((channel - mean) / std) ** 3)

    return [mean, variance, skewness]


# =================================================
# Frame-Level Color Features
# =================================================

def extract_color_histogram(frame, color_space="HSV", bins=16):
    """
    Extract normalized color histogram from one frame.
    """
    if color_space == "RGB":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ranges = [0, 256]
    elif color_space == "HSV":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = [0, 256]
    else:
        raise ValueError("color_space must be RGB or HSV")

    features = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [bins], ranges)
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    return np.concatenate(features)


def extract_color_moments(frame, color_space="HSV"):
    """
    Extract color moments from one frame.
    """
    if color_space == "RGB":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif color_space == "HSV":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("color_space must be RGB or HSV")

    moments = []
    for ch in range(3):
        moments.extend(_compute_color_moments(img[:, :, ch]))

    return np.array(moments)


# =================================================
# Texture Features (Frame-Level)
# =================================================

def extract_glcm_features(gray_frame):
    """
    Extract GLCM texture properties.

    Properties:
    - Contrast
    - Dissimilarity
    - Homogeneity
    - Energy
    - Correlation
    """
    glcm = graycomatrix(
        gray_frame,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    props = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ]

    return np.array([graycoprops(glcm, p)[0, 0] for p in props])


def extract_lbp_features(gray_frame, radius=1, points=8):
    """
    Extract Local Binary Pattern (LBP) histogram.

    Uses 'uniform' LBP and returns a normalized histogram.
    Safely handles floating-point outputs from skimage.
    """
    lbp = local_binary_pattern(
        gray_frame,
        points,
        radius,
        method="uniform"
    )

    # Number of bins for uniform LBP
    n_bins = points + 2

    # Convert to integer and clip to valid range
    lbp = np.clip(lbp.astype(np.int32), 0, n_bins - 1)

    hist = np.bincount(
        lbp.ravel(),
        minlength=n_bins
    ).astype(np.float32)

    # Normalize histogram
    hist /= (hist.sum() + 1e-6)

    return hist



def extract_gabor_features(gray_frame):
    """
    Extract Gabor filter responses (mean & variance).
    """
    features = []
    for theta in [0, np.pi / 4]:
        for sigma in [1.0, 3.0]:
            kernel = cv2.getGaborKernel(
                (21, 21), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray_frame, cv2.CV_8UC3, kernel)
            features.append(filtered.mean())
            features.append(filtered.var())
    return np.array(features)


# =================================================
# Video-Level Feature Aggregation
# =================================================

def extract_video_features(
    video_path,
    color_space="HSV",
    bins=16,
    max_frames=None
):
    """
    Extract combined color + texture features from video.
    """
    cap = cv2.VideoCapture(video_path)

    color_hist_sum = None
    color_moment_sum = None
    glcm_sum = None
    lbp_sum = None
    gabor_sum = None

    frame_count = 0

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
        if max_frames and frame_count >= max_frames:
            break

    cap.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames read: {video_path}")

    features = np.concatenate([
        color_hist_sum / frame_count,
        color_moment_sum / frame_count,
        glcm_sum / frame_count,
        lbp_sum / frame_count,
        gabor_sum / frame_count
    ])

    return features


# =================================================
# Visualization (Optional)
# =================================================

def plot_color_histogram_from_features(hist_features, bins=16, color_space="HSV"):
    """
    Plot histogram using extracted features only.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed, skipping plot")
        return

    labels = ["R", "G", "B"] if color_space == "RGB" else ["H", "S", "V"]

    plt.figure(figsize=(8, 4))
    for i, lbl in enumerate(labels):
        plt.plot(
            hist_features[i * bins:(i + 1) * bins],
            label=lbl
        )

    plt.title(f"{color_space} Color Histogram")
    plt.xlabel("Bin")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =================================================
# Main (Self-Test)
# =================================================

def main():
    """
    Simple sanity test on dataset structure:
    dataset/
      ├── class_1_Basketball
      ├── class_2_Biking
      └── class_3_WalkingWithDog
    """
    print("🔍 Running feature_extraction self-test")

    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset"

    sample_video = None
    for cls in dataset.iterdir():
        if cls.is_dir():
            vids = list(cls.glob("*.mp4")) + list(cls.glob("*.avi"))
            if vids:
                sample_video = vids[0]
                break

    if sample_video is None:
        print("❌ No video found")
        return

    print(f"🎬 Sample video: {sample_video.name}")

    features = extract_video_features(
        str(sample_video),
        color_space="HSV",
        bins=16,
        max_frames=30
    )

    print("✅ Feature extraction successful")
    print(f"📐 Feature vector length: {features.shape[0]}")


if __name__ == "__main__":
    main()
