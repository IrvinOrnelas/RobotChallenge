"""
classes/vision/perception.py
Visual perception pipeline operating on synthetic camera images.

Implements two techniques required by the hackathon rubric:
  1. Obstacle detection — colour segmentation (red/brown → bounding boxes)
  2. Landmark detection — ArUco-style square marker detection (orange/cyan/lime)
"""
import numpy as np


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
