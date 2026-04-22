"""
classes/husky/husky.py
Husky skid-steer robot — kinematics model + entity
"""
import numpy as np
from utils import wrap_angle, clamp, norm2


class HuskyModel:
    """
    Pure kinematics model for the Husky 4-wheel skid-steer robot.
    Mirrors the JS HuskyModel exactly.
    """
    def __init__(self, r: float = 0.1651, B: float = 0.555, maxspeed: float = 1.0):
        self.r = r            # wheel radius (m)
        self.B = B            # track width (m)
        self.maxspeed = maxspeed
        self.s = 1.0          # slip factor

    def forward_kinematics(self, wr1, wr2, wl1, wl2):
        """Wheel speeds → body twist (v, w)."""
        r_avg = (wr1 + wr2) / 2.0
        l_avg = (wl1 + wl2) / 2.0
        v = (self.r / 4.0) * (wr1 + wr2 + wl1 + wl2) * self.s
        w = (self.r / (2.0 * self.B)) * (r_avg - l_avg)
        return float(v), float(w)

    def inverse_kinematics(self, v: float, w: float):
        """Body twist → wheel speeds with saturation."""
        wr = (v + w * self.B / 2.0) / self.r
        wl = (v - w * self.B / 2.0) / self.r
        mx = max(abs(wr), abs(wl))
        if mx > self.maxspeed:
            wr *= self.maxspeed / mx
            wl *= self.maxspeed / mx
        return float(wr), float(wl)

    def compute_velocity_command(self, pose, goal, kv: float = 0.5, kw: float = 2.0):
        """Proportional controller: pose + goal → (v, w)."""
        dx = goal[0] - pose[0]
        dy = goal[1] - pose[1]
        dist = norm2(dx, dy)
        ang_err = wrap_angle(np.arctan2(dy, dx) - pose[2])
        v = clamp(kv * dist, -self.maxspeed, self.maxspeed)
        w = clamp(kw * ang_err, -2.0, 2.0)
        return float(v), float(w)

    def integrate(self, pose, wr: float, wl: float, dt: float):
        """Mid-point Euler integration. pose = [x, y, theta]."""
        v, w = self.forward_kinematics(wr, wr, wl, wl)
        theta_mid = pose[2] + w * dt / 2.0
        x = pose[0] + v * np.cos(theta_mid) * dt
        y = pose[1] + v * np.sin(theta_mid) * dt
        theta = wrap_angle(pose[2] + w * dt)
        return np.array([x, y, theta]), v, w


class Husky:
    """
    Husky robot entity — holds state and wraps HuskyModel.
    """
    def __init__(self, pose=(0.5, -1.8, 0.0), r=0.1651, B=0.555, maxspeed=1.0):
        self.model = HuskyModel(r=r, B=B, maxspeed=maxspeed)
        self.pose = np.array(pose, dtype=float)   # [x, y, theta]
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.v_meas = 0.0
        self.w_meas = 0.0
        self.trail = []

    def step(self, dt: float, wr: float, wl: float):
        self.pose, self.v_meas, self.w_meas = self.model.integrate(self.pose, wr, wl, dt)
        self.trail.append(self.pose[:2].copy())
        if len(self.trail) > 500:
            self.trail.pop(0)

    def set_twist(self, v: float, w: float):
        """Set velocity command and compute wheel speeds."""
        self.v_cmd, self.w_cmd = v, w
        return self.model.inverse_kinematics(v, w)

    @property
    def x(self): return self.pose[0]
    @property
    def y(self): return self.pose[1]
    @property
    def theta(self): return self.pose[2]
