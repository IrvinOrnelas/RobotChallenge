"""
classes/vision/camera.py
Synthetic first-person camera simulator using ray-casting.
Renders an RGB image from a robot's 2D world-pose perspective.
"""
import numpy as np


# ArUco-style landmark anchors: (id, world_x, world_y, rgb_color)
LANDMARKS = [
    (0,  0.5,  2.9, (255, 140,   0)),   # orange — near start
    (1,  7.0,  2.9, (  0, 180, 255)),   # cyan   — mid corridor
    (2, 13.0,  2.9, (140, 255,   0)),   # lime   — work zone
]


class CameraSimulator:
    """
    Raycaster-based synthetic RGB camera.

    Parameters
    ----------
    fov_deg   : horizontal field of view in degrees (default 90°)
    max_range : maximum sensing distance in metres
    img_w, img_h : output image resolution (pixels)
    """

    # Map box id → BGR-like wall colour
    _BOX_COLORS = {
        'A':  (239,  68,  68),   # red
        'B':  (245, 158,  11),   # amber
        'C':  ( 16, 185, 129),   # emerald
        'B1': (139,  69,  19),   # brown obstacle
        'B2': (139,  69,  19),
        'B3': (139,  69,  19),
    }
    _ROBOT_COLOR = (255, 235,  59)  # yellow
    _WALL_COLOR  = ( 70,  70, 100)  # dark blue-grey

    def __init__(self, fov_deg: float = 90.0, max_range: float = 6.0,
                 img_w: int = 320, img_h: int = 180):
        self.fov       = np.radians(fov_deg)
        self.max_range = max_range
        self.W         = img_w
        self.H         = img_h
        self.horizon   = img_h // 2

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    # Lane line definitions: (world_y, rgb_color)
    LANE_LINES = [
        ( 0.0, (255, 255, 255)),   # center — white
        ( 0.5, (255, 210,   0)),   # left lateral — yellow
        (-0.5, (255, 210,   0)),   # right lateral — yellow
    ]
    # Camera height above floor for floor-projection (metres)
    _CAM_HEIGHT = 0.55

    def render(self, robot_pose, boxes, other_robots=None,
               altitude: float = 0.0, draw_lanes: bool = False) -> np.ndarray:
        """
        Render a synthetic RGB image from *robot_pose* = [x, y, theta].

        Parameters
        ----------
        robot_pose  : (x, y, theta) of the camera-carrying robot
        boxes       : list of Box objects (obstacle + stack boxes)
        other_robots: list of other robot entities (optional)
        altitude    : camera height in metres (0 = ground, 3 = max height)
        draw_lanes  : if True, project LANE_LINES onto the floor region

        Returns
        -------
        numpy uint8 array  (H, W, 3)
        """
        rx, ry, rth = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])

        # ── Altitude-dependent parameters ────────────────────────────────────
        _MAX_ALT   = 3.0
        alt_norm   = min(1.0, altitude / _MAX_ALT)   # 0 → ground, 1 → max height

        # Horizon moves UP as drone rises (more floor becomes visible below)
        #   altitude=0 → horizon at H×0.50
        #   altitude=3 → horizon at H×0.28
        horizon = int(self.H * (0.50 - alt_norm * 0.22))
        self.horizon = max(4, horizon)   # update for landmark drawing

        # Column height shrinks with altitude (objects look smaller from above)
        col_h_scale = max(0.12, 1.0 - alt_norm * 0.78)

        # ── Background ───────────────────────────────────────────────────────
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Sky (above horizon)
        img[:self.horizon, :] = [15, 15, 28]

        # Floor gradient (below horizon) — brightens slightly at high altitude
        ground_base = 35 + int(alt_norm * 18)
        floor_h     = self.H - self.horizon
        if floor_h > 0:
            for row in range(self.horizon, self.H):
                prox = (row - self.horizon) / floor_h
                v    = ground_base + int(prox * 12 * alt_norm)
                img[row, :] = [v, max(0, v - 3), max(0, v - 9)]

        other_robots = other_robots or []

        # ── Raycasting ───────────────────────────────────────────────────────
        for col in range(self.W):
            ray_angle = rth + self.fov * (col / self.W - 0.5)
            hit, dist, color = self._cast_ray(rx, ry, ray_angle, boxes, other_robots)

            if hit and dist > 0.01:
                col_h  = int(min(self.H * 2, self.H * 1.8 * col_h_scale / dist))
                top    = max(0, self.horizon - col_h // 2)
                bot    = min(self.H, self.horizon + col_h // 2)
                shade  = max(0.18, 1.0 - dist / self.max_range)
                shaded = tuple(int(c * shade) for c in color)
                img[top:bot, col] = shaded

        # ── Floor perspective grid (drone effect, visible above alt≈0.9 m) ──
        if alt_norm > 0.30:
            strength  = int(alt_norm * 20)
            grid_step = 20
            for gc in range(0, self.W, grid_step):
                img[self.horizon:, gc] = np.clip(
                    img[self.horizon:, gc].astype(np.int16) + strength, 0, 255)
            for gr in range(self.horizon, self.H, grid_step):
                img[gr, :] = np.clip(
                    img[gr, :].astype(np.int16) + strength, 0, 255)

        # ── Floor lane lines (perspective projection) ────────────────────────
        if draw_lanes:
            self._draw_floor_lanes(img, rx, ry, rth)

        # ── Landmarks ────────────────────────────────────────────────────────
        self._draw_landmarks(img, rx, ry, rth)

        # ── Horizon accent line ───────────────────────────────────────────────
        img[self.horizon, :] = np.clip(
            img[self.horizon, :].astype(np.int16) + 22, 0, 255)

        return img

    # ── INTERNAL HELPERS ─────────────────────────────────────────────────────

    def _draw_floor_lanes(self, img, rx, ry, rth):
        """
        Inverse-perspective project LANE_LINES onto the floor region of img.
        Each floor pixel's world position is computed analytically; pixels that
        fall on a lane stripe are recolored with perspective-correct shading.
        """
        focal = (self.W / 2.0) / np.tan(self.fov / 2.0)
        h_cam = self._CAM_HEIGHT
        cos_th, sin_th = np.cos(rth), np.sin(rth)

        cols = np.arange(self.W, dtype=np.float32)
        x_screen = cols - self.W / 2.0          # (W,) lateral pixel offset

        for r in range(self.horizon + 1, self.H):
            y_screen = r - self.horizon           # pixels below horizon
            depth = h_cam * focal / y_screen      # metres to floor point
            if depth > self.max_range:
                continue

            # World-space lateral offset for every column
            lateral = x_screen * (depth / focal)  # (W,)

            # World Y coordinate of each floor pixel in this row
            wy = ry + depth * sin_th + lateral * cos_th   # (W,)

            shade = float(np.clip(1.0 - depth / self.max_range, 0.25, 1.0))
            # Perspective-correct stripe half-width: 2 px projected to world
            tol = max(0.035, 2.0 * depth / focal)

            for lane_y, color in self.LANE_LINES:
                mask = np.abs(wy - lane_y) < tol
                if mask.any():
                    img[r, mask] = (
                        int(color[0] * shade),
                        int(color[1] * shade),
                        int(color[2] * shade),
                    )

    def _cast_ray(self, rx, ry, angle, boxes, other_robots):
        dx = np.cos(angle)
        dy = np.sin(angle)

        best_dist  = self.max_range
        best_color = None

        # Boxes
        for box in boxes:
            x0, x1 = box.x - box.w / 2, box.x + box.w / 2
            y0, y1 = box.y - box.h / 2, box.y + box.h / 2
            dist = self._ray_aabb(rx, ry, dx, dy, x0, y0, x1, y1)
            if dist is not None and dist < best_dist:
                best_dist  = dist
                best_color = self._BOX_COLORS.get(box.id, (180, 180, 180))

        # Other robots (approximate as 0.3 m squares)
        for rob in other_robots:
            dist = self._ray_aabb(rx, ry, dx, dy,
                                  rob.x - 0.3, rob.y - 0.3,
                                  rob.x + 0.3, rob.y + 0.3)
            if dist is not None and dist < best_dist:
                best_dist  = dist
                best_color = self._ROBOT_COLOR

        # World boundary walls (top & bottom at y ≈ ±3)
        for wx0, wy0, wx1, wy1 in [(-0.5, -3.1, 14.5, -2.9),
                                    (-0.5,  2.9, 14.5,  3.1)]:
            dist = self._ray_aabb(rx, ry, dx, dy, wx0, wy0, wx1, wy1)
            if dist is not None and dist < best_dist:
                best_dist  = dist
                best_color = self._WALL_COLOR

        hit = best_color is not None
        return hit, best_dist, (best_color if hit else (60, 60, 60))

    @staticmethod
    def _ray_aabb(rx, ry, dx, dy, x0, y0, x1, y1) -> 'float | None':
        """Slab-method ray vs AABB. Returns entry distance or None."""
        eps = 1e-9
        t_min, t_max = 0.0, 1e9

        for o, d, lo, hi in [(rx, dx, x0, x1), (ry, dy, y0, y1)]:
            if abs(d) < eps:
                if o < lo or o > hi:
                    return None
            else:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None

        return t_min if t_min >= 0.01 else None

    def _draw_landmarks(self, img, rx, ry, rth):
        """Overlay ArUco-style landmark pillars into the image."""
        for lm_id, lm_x, lm_y, lm_color in LANDMARKS:
            dx     = lm_x - rx
            dy     = lm_y - ry
            dist   = np.sqrt(dx * dx + dy * dy)
            if dist < 0.1 or dist > self.max_range:
                continue
            rel_angle = _wrap_angle(np.arctan2(dy, dx) - rth)
            if abs(rel_angle) > self.fov / 2 * 1.05:
                continue
            col = int((rel_angle / self.fov + 0.5) * self.W)
            if not (0 <= col < self.W):
                continue
            bar_w = max(3, int(8.0 / dist))
            bar_h = max(6, int(self.H * 0.35 / dist))
            top   = max(0, self.horizon - bar_h)
            bot   = min(self.H, self.horizon + bar_h // 3)
            c0    = max(0, col - bar_w // 2)
            c1    = min(self.W, col + bar_w // 2 + 1)
            img[top:bot, c0:c1] = lm_color
            # Small white cap so it reads as a marker
            img[top:top + 2, c0:c1] = (255, 255, 255)


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
