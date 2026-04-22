"""
classes/husky/lidar.py
2D LiDAR sensor — ray casting against rectangular obstacles
"""
import numpy as np


class LiDAR:
    """
    Simulated 2D LiDAR sensor.
    Casts rays from robot pose and returns range readings.
    """
    def __init__(self, n_rays: int = 360, max_range: float = 8.0,
                 fov: float = 2 * np.pi, noise_std: float = 0.01):
        self.n_rays = n_rays
        self.max_range = max_range
        self.fov = fov
        self.noise_std = noise_std

    def _ray_box_intersect(self, ox, oy, dx, dy, box):
        """
        Ray-AABB intersection. box = (x_min, y_min, x_max, y_max).
        Returns distance or max_range if no hit.
        """
        x0, y0, x1, y1 = box
        t_min, t_max = 0.0, self.max_range

        for axis_o, axis_d, b0, b1 in [(ox, dx, x0, x1), (oy, dy, y0, y1)]:
            if abs(axis_d) < 1e-9:
                if axis_o < b0 or axis_o > b1:
                    return self.max_range
            else:
                ta = (b0 - axis_o) / axis_d
                tb = (b1 - axis_o) / axis_d
                t_min = max(t_min, min(ta, tb))
                t_max = min(t_max, max(ta, tb))
                if t_min > t_max:
                    return self.max_range

        return t_min if t_min >= 0 else self.max_range

    def scan(self, pose, obstacles: list):
        """
        Scan from pose=[x, y, theta].
        obstacles: list of dicts with keys x, y, w, h (center-based).
        Returns array of range readings (n_rays,).
        """
        x, y, theta = pose
        angles = theta + np.linspace(-self.fov / 2, self.fov / 2, self.n_rays)
        ranges = np.full(self.n_rays, self.max_range)

        boxes = []
        for ob in obstacles:
            bx, by, bw, bh = ob['x'], ob['y'], ob['w'], ob['h']
            boxes.append((bx - bw/2, by - bh/2, bx + bw/2, by + bh/2))

        for i, angle in enumerate(angles):
            dx, dy = np.cos(angle), np.sin(angle)
            for box in boxes:
                d = self._ray_box_intersect(x, y, dx, dy, box)
                ranges[i] = min(ranges[i], d)

        # Add Gaussian noise
        if self.noise_std > 0:
            ranges += np.random.normal(0, self.noise_std, self.n_rays)
            ranges = np.clip(ranges, 0, self.max_range)

        return ranges, angles

    def get_points(self, pose, ranges, angles):
        """Convert range/angle scan to 2D point cloud (world frame)."""
        x, y, _ = pose
        xs = x + ranges * np.cos(angles)
        ys = y + ranges * np.sin(angles)
        return np.column_stack([xs, ys])

    def detect_centroid(self, pose, ranges, angles, threshold=None):
        """
        Returns centroid of detected points closer than threshold.
        Useful for box detection.
        """
        if threshold is None:
            threshold = self.max_range * 0.8
        mask = ranges < threshold
        if not np.any(mask):
            return None
        pts = self.get_points(pose, ranges[mask], angles[mask])
        return pts.mean(axis=0)
