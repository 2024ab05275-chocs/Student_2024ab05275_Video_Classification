import cv2
import numpy as np

# =========================
# FEATURE EXTRACTION FUNCTIONS
# =========================

def extract_color_histogram(frame, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_hog_features(
    frame,
    win_size=(64, 128),
    block_size=(16, 16),
    block_stride=(8, 8),
    cell_size=(8, 8),
    nbins=9
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FORCE exact HOG window size
    if gray.shape[1] != win_size[0] or gray.shape[0] != win_size[1]:
        gray = cv2.resize(gray, win_size)

    hog = cv2.HOGDescriptor(
        win_size,
        block_size,
        block_stride,
        cell_size,
        nbins
    )

    features = hog.compute(gray)
    if features is None:
        return None

    return features.flatten()


def extract_frame_difference(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(curr_gray, prev_gray)

    hist = cv2.calcHist([diff], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extract_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mag_hist = cv2.calcHist(
        [mag.astype(np.float32)], [0], None,
        [16], [0, 20]
    )
    mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()

    ang_hist = cv2.calcHist(
        [ang.astype(np.float32)], [0], None,
        [16], [0, 2 * np.pi]
    )
    ang_hist = cv2.normalize(ang_hist, ang_hist).flatten()

    return np.concatenate([mag_hist, ang_hist])


# =========================
# TEMPORAL AGGREGATION (SAFE)
# =========================

def temporal_stats(features_list):
    """
    SAFE temporal aggregation:
    mean, std, min, max
    Handles variable-length features
    """
    if len(features_list) == 0:
        return None

    # Take reference length
    ref_len = len(features_list[0])
    clean_feats = []

    for f in features_list:
        if f is not None and len(f) == ref_len:
            clean_feats.append(f)

    if len(clean_feats) == 0:
        return None

    feats = np.vstack(clean_feats)   # <-- THIS FIXES YOUR ERROR

    mean = feats.mean(axis=0)
    std  = feats.std(axis=0)
    min_ = feats.min(axis=0)
    max_ = feats.max(axis=0)

    return np.concatenate([mean, std, min_, max_])



# =========================
# MAIN VIDEO FEATURE FUNCTION
# =========================

def extract_video_features(video_path, skip_frames=5, resize=(320, 240)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    color_feats = []
    hog_feats = []
    motion_diff_feats = []
    motion_flow_feats = []

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Empty video")

    prev_frame = cv2.resize(prev_frame, resize)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame = cv2.resize(frame, resize)

        if frame_idx % skip_frames == 0:
            # Color
            color_feats.append(
                extract_color_histogram(frame)
            )

            # HOG
            hog_feat = extract_hog_features(frame)
            if hog_feat is not None:
                hog_feats.append(hog_feat)

            # Motion
            motion_diff_feats.append(
                extract_frame_difference(prev_frame, frame)
            )
            motion_flow_feats.append(
                extract_optical_flow(prev_frame, frame)
            )

        # ALWAYS update previous frame
        prev_frame = frame

    cap.release()

    # Temporal aggregation
    color_vec = temporal_stats(color_feats)
    hog_vec = temporal_stats(hog_feats)
    diff_vec = temporal_stats(motion_diff_feats)
    flow_vec = temporal_stats(motion_flow_feats)

    if any(v is None for v in [color_vec, hog_vec, diff_vec, flow_vec]):
        raise ValueError(f"Insufficient valid frames in video: {video_path}")

    video_feature = np.concatenate([
        color_vec,
        hog_vec,
        diff_vec,
        flow_vec
    ])

    return video_feature


# =========================
# DATASET-LEVEL EXTRACTION
# =========================

def extract_features_for_split(split_data):
    """
    split_data: list of (video_path, label)
    """
    X, y = [], []

    for video_path, label in split_data:
        try:
            feat = extract_video_features(str(video_path))
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {video_path}: {e}")

    return np.array(X), np.array(y)


# =========================
# DEBUG / TEST
# =========================

if __name__ == "__main__":
    video_path = "sample_video.mp4"
    features = extract_video_features(video_path)
    print("✅ Feature vector shape:", features.shape)
