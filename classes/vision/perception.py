"""
classes/vision/perception.py
Visual perception pipeline operating on synthetic camera images.

Implements techniques required by the hackathon rubric:
  1. Obstacle detection — colour segmentation (red/brown → bounding boxes)
  2. Landmark detection — ArUco-style square marker detection (orange/cyan/lime)
  3. HoughFeatureExtractor — 14 Hough-segment features (mirrors BOOST.ipynb)
  4. GradientBoostingSteering — GB regressor for angular velocity correction
  5. LandmarkLocalizer — robot pose estimation via landmark triangulation
"""
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA


class PerceptionPipeline:
    """
    Stateless perception pipeline that operates on uint8 RGB images.
    All methods return (detections, annotated_image).
    """

    # ── OBSTACLE DETECTION ───────────────────────────────────────────────────

    def detect_obstacles(self, img: np.ndarray):
        """
        Colour-segment obstacle regions (red boxes, brown obstacles).

        Returns
        -------
        boxes     : list of (cx, cy, w, h) in pixel coords
        annotated : copy of img with green bounding boxes drawn
        """
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)

        # Red stack box A (shading can drop to ~45 at max range)
        red_mask   = (r > 45) & (r > g * 2) & (b < 50)
        # Amber stack box B
        amber_mask = (r > 80) & (g > 40) & (g < r) & (b < 30) & (r > b * 3)
        # Brown obstacle boxes B1/B2/B3 (shaded: r≈70-139, g≈25-69, b<25)
        brown_mask = (r > 55) & (r < 175) & (g > 20) & (g < 85) & (b < 25) & (r > g * 1.3)

        obs_mask = red_mask | amber_mask | brown_mask

        boxes = _find_column_bboxes(obs_mask, min_area=15)

        annotated = img.copy()
        for cx, cy, w, h in boxes:
            _draw_rect(annotated, cx - w // 2, cy - h // 2,
                       cx + w // 2, cy + h // 2, (0, 255, 0), thickness=2)

        return boxes, annotated

    # ── LANDMARK DETECTION ───────────────────────────────────────────────────

    def detect_landmarks(self, img: np.ndarray):
        """
        Detect ArUco-style coloured landmark pillars.

        Returns
        -------
        landmarks : list of (id, cx, cy, est_dist_m)
        annotated : copy of img with cross + ID annotation
        """
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)

        channels = [
            # id,  mask expression,                                   colour
            (0, (r > 200) & (g > 100) & (g < 180) & (b < 60),        (255, 140,   0)),   # orange
            (1, (r < 60)  & (g > 120) & (b > 180),                   (  0, 180, 255)),   # cyan
            (2, (r > 100) & (r < 190) & (g > 200) & (b < 60),        (140, 255,   0)),   # lime
        ]

        landmarks = []
        annotated  = img.copy()

        for lm_id, mask, color in channels:
            boxes = _find_column_bboxes(mask, min_area=4)
            if not boxes:
                continue
            # Take the tallest detection (most likely the pillar, not noise)
            cx, cy, w, h = max(boxes, key=lambda b: b[3])
            est_dist = max(0.3, 12.0 / max(1, h))   # apparent height → distance
            landmarks.append((lm_id, int(cx), int(cy), round(est_dist, 2)))

            # Cross annotation
            _draw_cross(annotated, cx, cy, size=6, color=color)
            # Small ID label (white 2×2 pixel block)
            lx = min(img.shape[1] - 4, cx + 5)
            ly = max(2, cy - 5)
            annotated[ly:ly + 2, lx:lx + 2] = (255, 255, 255)

        return landmarks, annotated

    # ── COMBINED ─────────────────────────────────────────────────────────────

    def annotate(self, img: np.ndarray):
        """
        Run both pipelines and return a merged annotated image.

        Returns
        -------
        annotated  : RGB uint8 image with all overlays
        obstacles  : list of (cx, cy, w, h)
        landmarks  : list of (id, cx, cy, est_dist_m)
        """
        obstacles, img_obs = self.detect_obstacles(img)
        landmarks, img_lm  = self.detect_landmarks(img)
        merged = np.maximum(img_obs, img_lm)
        return merged, obstacles, landmarks


# ── HELPERS (module-level, not methods) ──────────────────────────────────────

def _find_column_bboxes(mask: np.ndarray, min_area: int = 20):
    """
    Simple column-projection bounding-box extractor.
    Finds horizontal segments of non-zero columns and returns one bbox per segment.
    """
    if not mask.any():
        return []

    col_proj = mask.sum(axis=0)
    boxes    = []
    in_seg   = False
    c_start  = 0

    for c, val in enumerate(col_proj):
        if val > 0 and not in_seg:
            c_start, in_seg = c, True
        elif val == 0 and in_seg:
            _add_seg(mask, c_start, c, boxes, min_area)
            in_seg = False
    if in_seg:
        _add_seg(mask, c_start, len(col_proj), boxes, min_area)

    return boxes


def _add_seg(mask, c0, c1, boxes, min_area):
    sub      = mask[:, c0:c1]
    row_proj = sub.sum(axis=1)
    r_nz     = np.where(row_proj > 0)[0]
    if len(r_nz) == 0:
        return
    r0, r1 = int(r_nz[0]), int(r_nz[-1])
    w, h   = c1 - c0, r1 - r0 + 1
    if w * h < min_area:
        return
    boxes.append((int((c0 + c1) // 2), int((r0 + r1) // 2), w, h))


def _draw_rect(img, x0, y0, x1, y1, color, thickness=1):
    H, W = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    t = thickness
    img[y0:y0 + t, x0:x1 + 1]   = color   # top
    img[y1:y1 + t, x0:x1 + 1]   = color   # bottom
    img[y0:y1 + 1, x0:x0 + t]   = color   # left
    img[y0:y1 + 1, x1:x1 + t]   = color   # right


def _draw_cross(img, cx, cy, size, color):
    H, W = img.shape[:2]
    r0, r1 = max(0, cy - size), min(H, cy + size + 1)
    c0, c1 = max(0, cx - size), min(W, cx + size + 1)
    img[r0:r1, cx]  = color
    img[cy,    c0:c1] = color


# ── HOUGH FEATURE EXTRACTION ─────────────────────────────────────────────────

class HoughFeatureExtractor:
    """
    Extract 14 Hough-segment features from a camera image.
    Mirrors the feature pipeline in BOOST.ipynb (cells 8-11).
    """

    _FEATURE_KEYS = [
        'n_lines', 'mean_angle', 'std_angle', 'mean_abs_angle',
        'mean_length', 'max_length', 'left_count', 'right_count',
        'balance_lr', 'center_bottom', 'center_error',
        'vanishing_x', 'vanishing_error', 'confidence',
    ]

    def extract(self, img: np.ndarray, horizon: int = None) -> dict:
        """
        img      : uint8 RGB (H, W, 3)
        horizon  : floor starts at this row; defaults to H//2
        Returns dict with 14 float features + 'segments' list (not fed to GB).
        """
        H, W = img.shape[:2]
        if horizon is None:
            horizon = H // 2

        gray = rgb2gray(img)
        floor = np.zeros_like(gray)
        floor[horizon:] = gray[horizon:]

        edges = canny(floor, sigma=1.2, low_threshold=0.08, high_threshold=0.20)
        segments = probabilistic_hough_line(
            edges, threshold=12, line_length=18, line_gap=6)

        n = len(segments)
        if n == 0:
            return self._zero_features()

        angles, lengths = [], []
        left_count = right_count = 0
        cx_w = W / 2.0

        for (x0, y0), (x1, y1) in segments:
            angles.append(np.arctan2(y1 - y0, x1 - x0))
            lengths.append(np.hypot(x1 - x0, y1 - y0))
            if (x0 + x1) / 2.0 < cx_w:
                left_count += 1
            else:
                right_count += 1

        angles  = np.array(angles)
        lengths = np.array(lengths)

        center_bottom  = self._estimate_center_bottom(segments, W, H)
        center_error   = (center_bottom - cx_w) / cx_w
        vanishing_x    = self._estimate_vanishing_x(segments, W)
        vanishing_error = (vanishing_x - cx_w) / cx_w
        confidence     = min(1.0, n / 15.0)

        return {
            'n_lines':        float(n),
            'mean_angle':     float(np.mean(angles)),
            'std_angle':      float(np.std(angles)),
            'mean_abs_angle': float(np.mean(np.abs(angles))),
            'mean_length':    float(np.mean(lengths)),
            'max_length':     float(np.max(lengths)),
            'left_count':     float(left_count),
            'right_count':    float(right_count),
            'balance_lr':     float(left_count - right_count) / max(1, n),
            'center_bottom':  float(center_bottom),
            'center_error':   float(center_error),
            'vanishing_x':    float(vanishing_x),
            'vanishing_error': float(vanishing_error),
            'confidence':     float(confidence),
            'segments':       segments,
        }

    def to_vector(self, feat: dict) -> np.ndarray:
        """Return the 14 numeric features as a 1-D float32 array (excludes 'segments')."""
        return np.array([feat[k] for k in self._FEATURE_KEYS], dtype=np.float32)

    def _zero_features(self):
        d = {k: 0.0 for k in self._FEATURE_KEYS}
        d['segments'] = []
        return d

    def _estimate_center_bottom(self, segments, W, H):
        xs = []
        for (x0, y0), (x1, y1) in segments:
            if abs(y1 - y0) < 1e-3:
                continue
            x_bot = x0 + (x1 - x0) * (H - y0) / (y1 - y0)
            if 0 <= x_bot <= W:
                xs.append(x_bot)
        return float(np.mean(xs)) if xs else W / 2.0

    def _estimate_vanishing_x(self, segments, W):
        xs = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                (x0, y0), (x1, y1) = segments[i]
                (x2, y2), (x3, y3) = segments[j]
                denom = (x0 - x1) * (y2 - y3) - (y0 - y1) * (x2 - x3)
                if abs(denom) < 1e-6:
                    continue
                t = ((x0 - x2) * (y2 - y3) - (y0 - y2) * (x2 - x3)) / denom
                ix = x0 + t * (x1 - x0)
                if 0 <= ix <= W:
                    xs.append(ix)
        return float(np.mean(xs)) if xs else W / 2.0


# ── GRADIENT BOOSTING STEERING ───────────────────────────────────────────────

class GradientBoostingSteering:
    """
    GradientBoostingRegressor that predicts angular velocity (omega) corrections
    from Hough-segment features extracted from the robot's forward camera.

    Hyperparameters mirror BOOST.ipynb cells 16-19:
      n_estimators=120, learning_rate=0.08, max_depth=3
    Safety filter mirrors BOOST.ipynb cell 40:
      omega_safe = clip(0.65*prev + 0.35*pred, -0.8, 0.8)
    """

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.08,
            max_depth=3, random_state=42)
        self._trained    = False
        self._omega_prev = 0.0
        self._extractor  = HoughFeatureExtractor()

    def train_on_synthetic(self, camera, boxes, n_samples: int = 600):
        """
        Generate synthetic training frames from random corridor poses and fit the model.
        Ground-truth formula from BOOST.ipynb:
          omega = -1.25 * lat_error - 0.85 * head_error + N(0, 0.035)
        """
        rng = np.random.default_rng(42)
        X, y = [], []

        for _ in range(n_samples):
            px  = rng.uniform(0.5, 12.0)
            py  = rng.uniform(-1.5, 1.5)
            pth = rng.uniform(-np.pi / 4, np.pi / 4)

            img  = camera.render([px, py, pth], boxes, draw_lanes=True)
            feat = self._extractor.extract(img, horizon=img.shape[0] // 2)
            vec  = self._extractor.to_vector(feat)

            lat_error  = py / 1.5
            head_error = pth / (np.pi / 4)
            omega_true = (-1.25 * lat_error - 0.85 * head_error
                          + rng.normal(0, 0.035))
            X.append(vec)
            y.append(omega_true)

        self.model.fit(np.array(X), np.array(y))
        self._trained = True
        print(f"[GB] Trained on {n_samples} synthetic samples.")

    def predict(self, img: np.ndarray, horizon: int = None) -> float:
        """
        Extract Hough features from img and return a safety-filtered omega correction.
        Returns 0.0 if model is not yet trained.
        """
        if not self._trained:
            return 0.0
        feat      = self._extractor.extract(img, horizon=horizon)
        vec       = self._extractor.to_vector(feat).reshape(1, -1)
        omega_raw = float(self.model.predict(vec)[0])
        omega_safe = float(np.clip(
            0.65 * self._omega_prev + 0.35 * omega_raw, -0.8, 0.8))
        self._omega_prev = omega_safe
        return omega_safe

    def reset(self):
        self._omega_prev = 0.0


# ── LANDMARK LOCALIZER ───────────────────────────────────────────────────────

# World XY positions of the three ArUco-style landmarks (from camera.py LANDMARKS)
_LANDMARK_WORLD = {0: (0.5, 2.9), 1: (7.0, 2.9), 2: (13.0, 2.9)}


class LandmarkLocalizer:
    """
    Estimate robot XY position from detected ArUco-style landmark pillars.
    Uses bearing + distance from each visible landmark; weighted average when
    multiple landmarks are visible. Heading (theta) is unchanged from prior.
    """

    def estimate_pose(self, landmarks, prior_pose,
                      fov_rad: float, img_w: int) -> np.ndarray:
        """
        landmarks  : list of (id, cx_px, cy_px, est_dist_m) from PerceptionPipeline
        prior_pose : [x, y, theta] odometry estimate (fallback + heading prior)
        fov_rad    : camera horizontal FOV in radians
        img_w      : image width in pixels
        Returns [x_est, y_est, theta] — theta unchanged from prior_pose.
        """
        if not landmarks:
            return np.array(prior_pose, dtype=float)

        pth = float(prior_pose[2])
        estimates = []

        for lm_id, cx, cy, est_dist in landmarks:
            if lm_id not in _LANDMARK_WORLD:
                continue
            lm_wx, lm_wy = _LANDMARK_WORLD[lm_id]
            bearing = pth + fov_rad * (cx / img_w - 0.5)
            rx_est  = lm_wx - est_dist * np.cos(bearing)
            ry_est  = lm_wy - est_dist * np.sin(bearing)
            weight  = 1.0 / max(0.1, est_dist)
            estimates.append((rx_est, ry_est, weight))

        if not estimates:
            return np.array(prior_pose, dtype=float)

        total_w = sum(w for _, _, w in estimates)
        rx = sum(x * w for x, _, w in estimates) / total_w
        ry = sum(y * w for _, y, w in estimates) / total_w
        return np.array([rx, ry, pth], dtype=float)


# ── PCA BOX ORIENTATION ───────────────────────────────────────────────────────

class PCABoxOrientation:
    """
    Estimate the principal orientation of a detected obstacle box from its
    pixel mask using PCA. Returns the push angle (perpendicular to principal axis).
    """

    def estimate(self, img: np.ndarray,
                 bbox_cx: int, bbox_cy: int,
                 bbox_w: int, bbox_h: int) -> 'float | None':
        """
        img    : uint8 RGB image
        bbox_* : pixel bounding box of detected obstacle (center + size)
        Returns push angle in radians (world-space rotation suggestion),
                or None if fewer than 6 obstacle pixels found in bbox.
        """
        r = img[:, :, 0].astype(np.int16)
        g = img[:, :, 1].astype(np.int16)
        b = img[:, :, 2].astype(np.int16)

        y0 = max(0, bbox_cy - bbox_h // 2)
        y1 = min(img.shape[0], bbox_cy + bbox_h // 2)
        x0 = max(0, bbox_cx - bbox_w // 2)
        x1 = min(img.shape[1], bbox_cx + bbox_w // 2)

        r_roi = r[y0:y1, x0:x1]
        g_roi = g[y0:y1, x0:x1]
        b_roi = b[y0:y1, x0:x1]
        mask  = ((r_roi > 55) & (r_roi < 175) &
                 (g_roi > 20) & (g_roi < 85)  &
                 (b_roi < 25))

        ys, xs = np.where(mask)
        if len(xs) < 6:
            return None

        pts = np.column_stack([xs + x0, ys + y0]).astype(np.float32)
        pca = PCA(n_components=2)
        pca.fit(pts)
        principal = pca.components_[0]
        angle = np.arctan2(float(principal[1]), float(principal[0]))
        return float(angle + np.pi / 2)   # perpendicular = push direction
