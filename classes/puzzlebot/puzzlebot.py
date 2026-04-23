"""
classes/puzzlebot/puzzlebot.py
PuzzleBot differential-drive base + full robot entity
"""
import numpy as np
from utils import wrap_angle, clamp, norm2
import matplotlib
import matplotlib.patches as patches
from classes.puzzlebot.puzzlebot_arm import PuzzleBotArm, PuzzleBotArmModel


class PuzzleBotModel:
    """
    Differential-drive kinematics for PuzzleBot mobile base.
    """
    def __init__(self, r: float = 0.05, L: float = 0.19, maxspeed: float = 0.8):
        self.r = r          # wheel radius (m)
        self.L = L          # wheel separation (m)
        self.maxspeed = maxspeed

    def forward_kinematics(self, wr: float, wl: float):
        """Wheel speeds → (v, w)."""
        v = (self.r / 2.0) * (wr + wl)
        w = (self.r / self.L) * (wr - wl)
        return float(v), float(w)

    def inverse_kinematics(self, v: float, w: float):
        """(v, w) → wheel speeds with saturation."""
        wr = (2.0 * v + w * self.L) / (2.0 * self.r)
        wl = (2.0 * v - w * self.L) / (2.0 * self.r)
        mx = max(abs(wr), abs(wl))
        if mx > self.maxspeed:
            wr *= self.maxspeed / mx
            wl *= self.maxspeed / mx
        return float(wr), float(wl)

    def integrate(self, pose: np.ndarray, v: float, w: float, dt: float):
        """Euler integration. pose = [x, y, theta]."""
        x = pose[0] + v * np.cos(pose[2]) * dt
        y = pose[1] + v * np.sin(pose[2]) * dt
        theta = wrap_angle(pose[2] + w * dt)
        return np.array([x, y, theta])


class PuzzleBot:
    """
    Full PuzzleBot entity: differential base + 3-DOF arm.
    FSM states: IDLE → MOVING → GRASPING → STACKING → DONE
    """
    FSM_STATES = ['IDLE', 'MOVING', 'GRASPING', 'STACKING', 'DONE']

    def __init__(self, robot_id: int, pose=(0.0, 0.0, 0.0),
                 base_model: PuzzleBotModel = None,
                 arm: PuzzleBotArm = None):
        self.id = robot_id
        self.model = base_model or PuzzleBotModel()
        self.arm = arm or PuzzleBotArm()
        self.pose = np.array(pose, dtype=float)
        self.trail = []
        self.state = 'IDLE'
        self.target_box = None

    def set_twist(self, v: float, w: float, dt: float):
        """Apply velocity command for one timestep."""
        self.pose = self.model.integrate(self.pose, v, w, dt)
        self.trail.append(self.pose[:2].copy())
        if len(self.trail) > 500:
            self.trail.pop(0)

    def navigate_to(self, goal, dt: float, kv: float = 0.3, kw: float = 1.2):
        """
        Simple proportional controller toward a goal [x, y].
        Returns True when within tolerance.
        """
        dx = goal[0] - self.pose[0]
        dy = goal[1] - self.pose[1]
        dist = norm2(dx, dy)
        if dist < 0.05:
            return True
        ang_err = wrap_angle(np.arctan2(dy, dx) - self.pose[2])
        v = clamp(kv * dist, 0.0, 0.5)
        w = clamp(kw * ang_err, -1.5, 1.5)
        self.set_twist(v, w, dt)
        return False

    def step(self, dt: float):
        """Advance arm joints toward target."""
        self.arm.step(dt)

    @property
    def x(self): return self.pose[0]
    @property
    def y(self): return self.pose[1]
    @property
    def theta(self): return self.pose[2]

    def __repr__(self):
        return (f"PuzzleBot(id={self.id}, state={self.state}, "
                f"pos=({self.x:.3f},{self.y:.3f}), arm_q={[f'{q:.2f}' for q in self.arm.q]})")
        
    def setup_visuals(self, ax):
        """Setup Matplotlib artists for the PuzzleBot base and its arm."""
        colors = ['#fbbf24', '#34d399', '#f472b6']
        self.color = colors[self.id % 3] # Assign color based on ID

        # Trail
        self.trail_line, = ax.plot([], [], '-', color=self.color, lw=0.8, alpha=0.4, zorder=2)
        
        # Body
        self.body_patch = patches.Circle((0, 0), 0.15, fc=self.color, ec='white', lw=1, zorder=5)
        ax.add_patch(self.body_patch)
        
        # Label
        self.label = ax.text(0, 0, f'PB{self.id}', color=self.color, 
                             fontsize=6, fontfamily='monospace', zorder=6)

        artists = [self.trail_line, self.body_patch, self.label]
        artists.extend(self.arm.setup_visuals(ax))
        return artists

    def update_visuals(self, ax):
        """Update visual state per frame."""
        if len(self.trail) > 1:
            tr = np.array(self.trail)
            self.trail_line.set_data(tr[:,0], tr[:,1])

        self.body_patch.center = (self.x, self.y)
        self.label.set_position((self.x + 0.18, self.y + 0.18))

        # Pass base pose to the arm so it can draw itself in the world frame
        self.arm.update_visuals(self.x, self.y, self.theta)
