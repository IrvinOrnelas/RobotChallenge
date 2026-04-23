import numpy as np
import matplotlib.patches as patches

class Zone:
    """Defines a rectangular area and handles point-inclusion checks."""
    def __init__(self, name, x_min, x_max, y_min, y_max, color, text_offset=(0.2, 0.1)):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.color = color
        self.text_offset = text_offset

    def get_points_inside(self, points):
        """Filters an Nx2 array of points, returning only those inside the zone."""
        if len(points) == 0:
            return np.array([])
        
        mask = (points[:, 0] > self.x_min) & (points[:, 0] < self.x_max) & \
               (points[:, 1] > self.y_min) & (points[:, 1] < self.y_max)
        return points[mask]

    def setup_visuals(self, ax):
        """Draws the zone in the simulator."""
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        self.patch = patches.Rectangle(
            (self.x_min, self.y_min), w, h,
            fc=self.color, ec=self.color, lw=1, ls='--', alpha=0.15, zorder=1
        )
        ax.add_patch(self.patch)
        self.text = ax.text(
            self.x_min + self.text_offset[0], self.y_min + self.text_offset[1],
            self.name, color=self.color, fontsize=7, fontfamily='monospace', zorder=2
        )
        return [self.patch, self.text]

    def update_visuals(self):
        """Static zones don't move, but this maintains the visual update pattern."""
        pass
    
    def contains(self, x, y):
        """Returns True if the point (x,y) is strictly inside the zone boundaries."""
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)