"""
classes/planning/astar.py
A* path planner on a 2D occupancy grid.

World: 14 × 6 m  (x: 0→14, y: -3→3)
Resolution: 0.1 m → 140 × 60 cells
"""
import heapq
import numpy as np


class AStarPlanner:
    """
    Grid-based A* planner with dynamic replanning support.

    Parameters
    ----------
    world_w, world_h : world dimensions in metres
    resolution       : grid cell size in metres
    """

    def __init__(self, world_w: float = 14.0, world_h: float = 6.0,
                 resolution: float = 0.1):
        self.res  = resolution
        self.W    = world_w
        self.H    = world_h
        self.cols = int(world_w / resolution)
        self.rows = int(world_h / resolution)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.replan_count = 0
        self._current_path: list = []

    # ── GRID CONSTRUCTION ────────────────────────────────────────────────────

    def build_grid(self, static_boxes, margin: float = 0.18):
        """Build occupancy grid from a list of Box objects."""
        self.grid[:] = 0
        # World walls
        self._mark_rect(-0.5, -3.2, 14.5, -2.85 + margin)
        self._mark_rect(-0.5,  2.85 - margin, 14.5, 3.2)

        for box in static_boxes:
            self._mark_rect(box.x - box.w / 2 - margin,
                             box.y - box.h / 2 - margin,
                             box.x + box.w / 2 + margin,
                             box.y + box.h / 2 + margin)

    def _mark_rect(self, x0, y0, x1, y1):
        c0 = max(0, int(x0 / self.res))
        c1 = min(self.cols, int(x1 / self.res) + 1)
        r0 = max(0, int((y0 + self.H / 2) / self.res))
        r1 = min(self.rows, int((y1 + self.H / 2) / self.res) + 1)
        self.grid[r0:r1, c0:c1] = 1

    # ── COORDINATE CONVERSION ────────────────────────────────────────────────

    def _xy_to_cell(self, x: float, y: float):
        c = int(x / self.res)
        r = int((y + self.H / 2) / self.res)
        return (int(np.clip(c, 0, self.cols - 1)),
                int(np.clip(r, 0, self.rows - 1)))

    def _cell_to_xy(self, c: int, r: int):
        return (c * self.res + self.res / 2,
                r * self.res + self.res / 2 - self.H / 2)

    # ── PLANNING ─────────────────────────────────────────────────────────────

    def plan(self, start_xy, goal_xy, dynamic_obs=None):
        """
        Compute collision-free path from start_xy to goal_xy.

        Parameters
        ----------
        start_xy, goal_xy : (x, y) in world metres
        dynamic_obs       : optional list of Box objects to add temporarily

        Returns
        -------
        List of (x, y) waypoints (world metres), including goal.
        Falls back to [goal_xy] if no path is found.
        """
        if dynamic_obs:
            for obs in dynamic_obs:
                self._mark_rect(obs.x - obs.w / 2 - 0.1,
                                 obs.y - obs.h / 2 - 0.1,
                                 obs.x + obs.w / 2 + 0.1,
                                 obs.y + obs.h / 2 + 0.1)

        sc = self._xy_to_cell(*start_xy)
        gc = self._xy_to_cell(*goal_xy)

        cells = self._astar(sc, gc)
        if not cells:
            self._current_path = [list(goal_xy)]
            return self._current_path

        waypoints = [self._cell_to_xy(c, r) for c, r in cells]
        waypoints = _smooth_waypoints(waypoints, step=4)
        waypoints.append(list(goal_xy))   # ensure exact goal
        self._current_path = waypoints
        return waypoints

    def replan(self, start_xy, goal_xy, new_obs_dicts=None):
        """
        Replan after camera detects new obstacles.

        Parameters
        ----------
        new_obs_dicts : list of {'x', 'y', 'w', 'h'} dicts from perception
        """
        self.replan_count += 1
        if new_obs_dicts:
            for obs in new_obs_dicts:
                self._mark_rect(obs['x'] - obs['w'] / 2 - 0.12,
                                 obs['y'] - obs['h'] / 2 - 0.12,
                                 obs['x'] + obs['w'] / 2 + 0.12,
                                 obs['y'] + obs['h'] / 2 + 0.12)
        return self.plan(start_xy, goal_xy)

    # ── A* SEARCH ────────────────────────────────────────────────────────────

    def _astar(self, start, goal):
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from  = {start: None}
        g_score    = {start: 0.0}

        def heuristic(cell):
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return _reconstruct(came_from, current)

            for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                nc, nr = current[0] + dc, current[1] + dr
                if not (0 <= nc < self.cols and 0 <= nr < self.rows):
                    continue
                if self.grid[nr, nc]:
                    continue
                move_cost = 1.0 if (dc == 0 or dr == 0) else 1.414
                ng = g_score[current] + move_cost
                neighbour = (nc, nr)
                if ng < g_score.get(neighbour, float('inf')):
                    g_score[neighbour] = ng
                    came_from[neighbour] = current
                    heapq.heappush(open_heap,
                                   (ng + heuristic(neighbour), neighbour))
        return []


# ── MODULE HELPERS ────────────────────────────────────────────────────────────

def _reconstruct(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    return list(reversed(path))


def _smooth_waypoints(waypoints, step: int = 4):
    """Downsample a dense cell-path to fewer waypoints."""
    if len(waypoints) <= 2:
        return waypoints
    return waypoints[::step]
