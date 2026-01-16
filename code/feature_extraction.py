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

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# =================================================
# LOGGING UTILITY
# =================================================


def log(message: str) -> None:
    """Centralized logging helper."""
    print(f"[INFO] {message}", flush=True)

# ============================================================
# CLASSICAL FEATURE VISUALIZATION SETUP
# ============================================================

CLASSICAL_VIZ_DIR = "results/feature_visualizations/classical"
os.makedirs(CLASSICAL_VIZ_DIR, exist_ok=True)

# =================================================
# COLOR FEATURE HELPERS
# =================================================
# This helper function computes color moments that
# summarize the statistical distribution of pixel
# intensities within a single image channel.
#
# Output:
#   [mean, variance, skewness]
# =================================================


def _compute_color_moments(channel: np.ndarray) -> List[float]:
    """
    Compute the first three statistical color moments
    for a single image channel.

    Moments captured:
      1. Mean      → Average intensity (brightness)
      2. Variance  → Intensity spread (contrast)
      3. Skewness  → Distribution asymmetry

    Args:
        channel (np.ndarray):
            2D array representing a single color channel
            (e.g., R, G, or B), values typically in [0, 255]

    Returns:
        List[float]:
            [mean, variance, skewness]
    """

    # -------------------------------------------------
    # Convert to float for numerical stability
    # -------------------------------------------------
    channel = channel.astype(np.float32)

    # -------------------------------------------------
    # First moment: Mean
    # -------------------------------------------------
    # Represents the average intensity of the channel
    mean = np.mean(channel)

    # -------------------------------------------------
    # Second moment: Variance
    # -------------------------------------------------
    # Measures the spread / contrast of intensities
    variance = np.var(channel)

    # -------------------------------------------------
    # Standard deviation (used for skewness normalization)
    # Add epsilon to avoid division by zero
    # -------------------------------------------------
    std = np.sqrt(variance) + 1e-6

    # -------------------------------------------------
    # Third moment: Skewness
    # -------------------------------------------------
    # Measures asymmetry of the intensity distribution
    # Positive skew → long right tail
    # Negative skew → long left tail
    skewness = np.mean(((channel - mean) / std) ** 3)

    return [mean, variance, skewness]



# =================================================
# FRAME-LEVEL COLOR FEATURE EXTRACTION
# =================================================
# This module extracts color-based appearance features
# from individual video frames using:
#   - Color histograms
#   - Color moments
#
# Input:
#   frame: np.ndarray of shape (H, W, 3), BGR format
# =================================================



def extract_color_histogram(
    frame: np.ndarray,
    color_space: str = "HSV",
    bins: int = 16
) -> np.ndarray:
    """
    Extract normalized color histograms from a video frame.

    For each color channel:
      - Compute histogram
      - Normalize
      - Concatenate across channels

    Args:
        frame (np.ndarray):
            Input image in BGR format (H, W, 3)
        color_space (str):
            Target color space: "RGB" or "HSV"
        bins (int):
            Number of histogram bins per channel

    Returns:
        np.ndarray:
            Concatenated color histogram of shape (3 * bins,)
    """

    # -------------------------------------------------
    # Sanity checks
    # -------------------------------------------------
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame shape (H, W, 3)")

    if color_space not in {"RGB", "HSV"}:
        raise ValueError("color_space must be 'RGB' or 'HSV'")

    # -------------------------------------------------
    # Convert from BGR (OpenCV default) to target space
    # -------------------------------------------------
    if color_space == "RGB":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    features = []

    # -------------------------------------------------
    # Compute histogram for each channel
    # -------------------------------------------------
    for ch in range(3):
        # Calculate histogram
        hist = cv2.calcHist(
            [image],
            [ch],
            None,
            [bins],
            [0, 256]
        )

        # Normalize histogram (L2 normalization)
        hist = cv2.normalize(hist, hist).flatten()

        features.append(hist)

    # -------------------------------------------------
    # Concatenate histograms across channels
    # -------------------------------------------------
    return np.concatenate(features)


def extract_color_moments(
    frame: np.ndarray,
    color_space: str = "HSV"
) -> np.ndarray:
    """
    Extract color moments from a video frame.

    For each channel:
      - Mean
      - Variance
      - Skewness

    Args:
        frame (np.ndarray):
            Input image in BGR format (H, W, 3)
        color_space (str):
            Target color space: "RGB" or "HSV"

    Returns:
        np.ndarray:
            Color moments vector of shape (9,)
    """

    # -------------------------------------------------
    # Sanity checks
    # -------------------------------------------------
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame shape (H, W, 3)")

    if color_space not in {"RGB", "HSV"}:
        raise ValueError("color_space must be 'RGB' or 'HSV'")

    # -------------------------------------------------
    # Convert from BGR to target color space
    # -------------------------------------------------
    if color_space == "RGB":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    moments = []

    # -------------------------------------------------
    # Compute color moments per channel
    # -------------------------------------------------
    for ch in range(3):
        moments.extend(
            _compute_color_moments(image[:, :, ch])
        )

    return np.array(moments, dtype=np.float32)



# =================================================
# TEXTURE FEATURE EXTRACTION
# =================================================
# This module extracts complementary texture descriptors
# from grayscale images using classical CV techniques:
#   - GLCM (statistical texture)
#   - LBP  (local micro-patterns)
#   - Gabor filters (frequency & orientation)
#
# Input convention:
#   gray: np.ndarray of shape (H, W), dtype uint8
# =================================================


def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    """
    Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).

    Computes second-order statistics that describe how
    pixel intensities co-occur in local neighborhoods.

    Args:
        gray (np.ndarray):
            Grayscale image of shape (H, W), values in [0, 255]

    Returns:
        np.ndarray:
            GLCM feature vector of shape (5,)
    """

    # -------------------------------------------------
    # Sanity checks
    # -------------------------------------------------
    if gray.ndim != 2:
        raise ValueError("GLCM requires a 2D grayscale image")

    # -------------------------------------------------
    # Compute GLCM
    # distance=1, angle=0° (horizontal neighbors)
    # symmetric=True → (i, j) == (j, i)
    # normed=True → probabilities sum to 1
    # -------------------------------------------------
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    # -------------------------------------------------
    # Texture properties to extract
    # -------------------------------------------------
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation"
    ]

    # -------------------------------------------------
    # Extract each property and flatten
    # Shape from graycoprops: (distances, angles)
    # -------------------------------------------------
    features = [
        graycoprops(glcm, prop)[0, 0]
        for prop in properties
    ]

    return np.array(features, dtype=np.float32)


def extract_lbp_features(
    gray: np.ndarray,
    radius: int = 1,
    points: int = 8
) -> np.ndarray:
    """
    Extract texture features using Uniform Local Binary Patterns (LBP).

    LBP encodes local pixel neighborhoods into discrete
    binary patterns that represent texture primitives.

    Args:
        gray (np.ndarray):
            Grayscale image of shape (H, W)
        radius (int):
            Radius of neighborhood
        points (int):
            Number of sampling points

    Returns:
        np.ndarray:
            Normalized LBP histogram of shape (points + 2,)
    """

    # -------------------------------------------------
    # Compute uniform LBP codes
    # -------------------------------------------------
    lbp = local_binary_pattern(
        gray,
        P=points,
        R=radius,
        method="uniform"
    )

    # -------------------------------------------------
    # Uniform LBP produces (points + 2) distinct labels
    # -------------------------------------------------
    n_bins = points + 2

    # Clip values to valid bin range
    lbp = np.clip(
        lbp.astype(np.int32),
        0,
        n_bins - 1
    )

    # -------------------------------------------------
    # Compute normalized histogram
    # -------------------------------------------------
    hist = np.bincount(
        lbp.ravel(),
        minlength=n_bins
    ).astype(np.float32)

    hist /= (hist.sum() + 1e-6)

    return hist


def extract_gabor_features(gray: np.ndarray) -> np.ndarray:
    """
    Extract texture features using Gabor filters.

    Gabor filters capture oriented and frequency-specific
    texture information similar to the human visual system.

    Returns mean and variance of filter responses.

    Args:
        gray (np.ndarray):
            Grayscale image of shape (H, W)

    Returns:
        np.ndarray:
            Gabor feature vector of shape (8,)
    """

    features = []

    # -------------------------------------------------
    # Apply filters at multiple orientations and scales
    # -------------------------------------------------
    for theta in (0, np.pi / 4):          # Orientations
        for sigma in (1.0, 3.0):          # Scales

            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=sigma,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0
            )

            # Apply Gabor filter
            filtered = cv2.filter2D(
                gray,
                cv2.CV_32F,
                kernel
            )

            # Aggregate filter response
            features.append(filtered.mean())
            features.append(filtered.var())

    return np.array(features, dtype=np.float32)



# =================================================
# MOTION FEATURE EXTRACTION
# =================================================
# These functions extract compact motion descriptors
# from frame-to-frame difference representations.
#
# Expected input:
#   diff: np.ndarray
#     - Motion intensity values
#     - Can be 1D or multi-dimensional
# =================================================



def extract_motion_statistics(diff: np.ndarray) -> np.ndarray:
    """
    Compute basic statistical descriptors of motion intensity.

    Captures:
      - Mean     : Average motion energy
      - Variance : Motion variability
      - Max      : Strongest motion event

    Args:
        diff (np.ndarray):
            Motion intensity array (frame differences or motion magnitudes)

    Returns:
        np.ndarray:
            Motion statistics vector of shape (3,)
    """

    # -------------------------------------------------
    # Convert to float for numerical stability
    # -------------------------------------------------
    diff = diff.astype(np.float32)

    # -------------------------------------------------
    # Compute statistical motion descriptors
    # -------------------------------------------------
    mean_motion = diff.mean()
    var_motion = diff.var()
    max_motion = diff.max()

    return np.array([
        mean_motion,
        var_motion,
        max_motion
    ], dtype=np.float32)


def extract_motion_histogram(diff: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Compute a normalized histogram of motion magnitudes.

    This captures the distribution of motion intensities
    rather than just aggregate statistics.

    Args:
        diff (np.ndarray):
            Motion intensity values (expected range: 0–255)
        bins (int):
            Number of histogram bins

    Returns:
        np.ndarray:
            Normalized motion histogram of shape (bins,)
    """

    # -------------------------------------------------
    # Convert to float for consistency
    # -------------------------------------------------
    diff = diff.astype(np.float32)

    # -------------------------------------------------
    # Compute histogram over motion magnitudes
    # -------------------------------------------------
    hist, bin_edges = np.histogram(
        diff,
        bins=bins,
        range=(0, 256)  # Standard range for 8-bit motion
    )

    # -------------------------------------------------
    # Normalize histogram to sum to 1
    # (Add epsilon to avoid division by zero)
    # -------------------------------------------------
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist



# =================================================
# TEMPORAL FEATURE EXTRACTION
# =================================================
# These functions convert a sequence of frame-level
# features into fixed-length temporal descriptors.
#
# Input shape convention:
#   sequence: (T, D)
#   T = number of frames
#   D = feature dimensionality per frame
# =================================================


def extract_temporal_statistics(sequence: np.ndarray) -> np.ndarray:
    """
    Compute statistical temporal descriptors over a feature sequence.

    This function summarizes how each feature dimension behaves
    across time by computing:
      - Mean  : Average activation (presence)
      - Std   : Temporal variability (motion / instability)
      - Min   : Lowest observed value
      - Max   : Highest observed value

    Args:
        sequence (np.ndarray):
            Frame-level feature sequence of shape (T, D)

    Returns:
        np.ndarray:
            Concatenated statistics vector of shape (4 * D,)
    """

    # -------------------------------------------------
    # Sanity checks
    # -------------------------------------------------
    if sequence.ndim != 2:
        raise ValueError(
            f"Expected input shape (T, D), got {sequence.shape}"
        )

    # -------------------------------------------------
    # Compute temporal statistics per feature dimension
    # -------------------------------------------------
    mean_features = sequence.mean(axis=0)
    std_features = sequence.std(axis=0)
    min_features = sequence.min(axis=0)
    max_features = sequence.max(axis=0)

    # -------------------------------------------------
    # Concatenate all statistics into a single vector
    # -------------------------------------------------
    temporal_stats = np.concatenate([
        mean_features,
        std_features,
        min_features,
        max_features
    ])

    return temporal_stats


def extract_temporal_gradients(sequence: np.ndarray) -> np.ndarray:
    """
    Compute first-order temporal gradient features.

    Temporal gradients capture frame-to-frame changes
    in the feature space, highlighting motion intensity
    and dynamic transitions.

    The gradient is computed as:
        gradient[t] = sequence[t+1] - sequence[t]

    Args:
        sequence (np.ndarray):
            Frame-level feature sequence of shape (T, D)

    Returns:
        np.ndarray:
            Global gradient statistics of shape (3,)
            [mean_gradient, std_gradient, max_gradient]
    """

    # -------------------------------------------------
    # Sanity checks
    # -------------------------------------------------
    if sequence.ndim != 2:
        raise ValueError(
            f"Expected input shape (T, D), got {sequence.shape}"
        )

    if sequence.shape[0] < 2:
        raise ValueError(
            "At least 2 frames are required to compute gradients"
        )

    # -------------------------------------------------
    # Compute frame-to-frame temporal gradients
    # Shape: (T - 1, D)
    # -------------------------------------------------
    gradients = np.diff(sequence, axis=0)

    # -------------------------------------------------
    # Aggregate gradient information globally
    # -------------------------------------------------
    mean_gradient = gradients.mean()
    std_gradient = gradients.std()
    max_gradient = gradients.max()

    return np.array([
        mean_gradient,
        std_gradient,
        max_gradient
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

# ============================================================
# CLASSICAL FEATURE IMPORTANCE
# ============================================================

def visualize_classical_feature_importance(
    model,
    feature_names,
    top_k=20,
    save_name="feature_importance.png"
):
    """
    Visualizes feature importance for classical ML models.

    Supports:
    - Tree-based models (Random Forest, Gradient Boosting)
    - Linear models (SVM, Logistic Regression)

    Args:
        model: trained classical ML model
        feature_names (list): names of handcrafted features
        top_k (int): number of top features to plot
        save_name (str): output image filename
    """

    # Extract importance scores
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        raise ValueError("Model does not support feature importance")

    # Sort features by importance
    indices = np.argsort(importance)[::-1][:top_k]

    plt.figure(figsize=(10, 6))
    plt.barh(
        range(top_k),
        importance[indices][::-1]
    )
    plt.yticks(
        range(top_k),
        np.array(feature_names)[indices][::-1]
    )
    plt.xlabel("Importance Score")
    plt.title("Top Classical Feature Importances")
    plt.tight_layout()

    save_path = os.path.join(CLASSICAL_VIZ_DIR, save_name)
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] Feature importance → {save_path}")

# ============================================================
# CLASSICAL FEATURE SPACE VISUALIZATION
# ============================================================

def visualize_classical_feature_space(
    features,
    labels,
    method="tsne",
    save_name="tsne.png"
):
    """
    Visualizes classical feature space using t-SNE or UMAP.

    Args:
        features (np.ndarray): shape (N, D)
        labels (np.ndarray): class labels
        method (str): 'tsne' or 'umap'
        save_name (str): output image filename
    """

    # Standardize features before dimensionality reduction
    features = StandardScaler().fit_transform(features)

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42
        )
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("Install umap-learn to use UMAP")
        reducer = umap.UMAP(
            n_components=2,
            random_state=42
        )
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=12
    )
    plt.colorbar(scatter)
    plt.title(f"{method.upper()} Visualization of Classical Features")
    plt.tight_layout()

    save_path = os.path.join(CLASSICAL_VIZ_DIR, save_name)
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] {method.upper()} plot → {save_path}")

# =================================================
# FEATURE NAMES HELPER
# =================================================

def get_classical_feature_names(
    hist_bins: int = 16,
    lbp_points: int = 8,
    include_motion: bool = True
) -> list[str]:
    """
    Generate feature names that exactly match the final
    concatenated video-level feature vector.
    """

    names = []

    # -----------------------
    # Per-frame raw features
    # -----------------------
    frame_feats = []

    # Color histogram (3 channels × bins)
    for ch in ("C1", "C2", "C3"):
        for b in range(hist_bins):
            frame_feats.append(f"{ch}_hist_bin{b}")

    # Color moments (3 channels × 3 moments)
    moments = ("mean", "var", "skew")
    for ch in ("C1", "C2", "C3"):
        for m in moments:
            frame_feats.append(f"{ch}_{m}")

    # Texture features
    glcm_props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    frame_feats.extend([f"glcm_{p}" for p in glcm_props])
    
    lbp_bins = lbp_points + 2
    frame_feats.extend([f"lbp_bin{b}" for b in range(lbp_bins)])
    
    gabor_feats = 8
    frame_feats.extend([f"gabor_f{i}" for i in range(gabor_feats)])

    # -----------------------
    # Temporal stats over frame features
    # -----------------------
    for stat in ("mean", "std", "min", "max"):
        names.extend([f"{f}_{stat}" for f in frame_feats])

    # -----------------------
    # Temporal gradients (global 3)
    # -----------------------
    names.extend(["grad_mean", "grad_std", "grad_max"])

    # -----------------------
    # Motion features (if available)
    # -----------------------
    if include_motion:
        motion_feats = ["motion_mean", "motion_var", "motion_max"]
        motion_hist = [f"motion_hist_bin{i}" for i in range(hist_bins)]
        motion_frame_feats = motion_feats + motion_hist

        # Temporal aggregation over motion features
        for stat in ("mean", "std", "min", "max"):
            names.extend([f"{f}_{stat}" for f in motion_frame_feats])

    return names



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
