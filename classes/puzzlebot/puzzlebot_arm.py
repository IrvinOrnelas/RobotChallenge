"""
classes/puzzlebot/puzzlebot_arm.py
3-DOF PuzzleBot arm — FK, IK, Jacobian, force-torque mapping
"""
import numpy as np
from utils import clamp, norm2


class PuzzleBotArmModel:
    """
    Geometric model of the 3-DOF revolute arm.
      q1 : base rotation (horizontal, full 360°)
      q2 : shoulder (±90°)
      q3 : elbow   (±90°)
    Link lengths: l1 (base height), l2 (upper arm), l3 (forearm)
    """
    def __init__(self, l1: float = 0.10, l2: float = 0.08, l3: float = 0.06):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.q_limits = [
            (-np.pi, np.pi),          # q1: full rotation
            (-np.pi/2, np.pi/2),      # q2: ±90°
            (-np.pi/2, np.pi/2),      # q3: ±90°
        ]

    # ── FORWARD KINEMATICS ──────────────────────────────────────────────────
    def forward_kinematics(self, q1: float, q2: float, q3: float) -> np.ndarray:
        """
        Joint angles → end-effector position [x, y, z].
        q1 controls horizontal rotation, q2/q3 vertical reach.
        """
        r = self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3)
        x = r * np.cos(q1)
        y = r * np.sin(q1)
        z = self.l1 + self.l2 * np.sin(q2) + self.l3 * np.sin(q2 + q3)
        return np.array([x, y, z])

    # ── INVERSE KINEMATICS ──────────────────────────────────────────────────
    def inverse_kinematics(self, x: float, y: float, z: float):
        """
        Target position → joint angles using geometric method.
        Returns (q1, q2, q3) or None if unreachable.
        """
        q1 = np.arctan2(y, x)
        r = norm2(x, y)
        zr = z - self.l1

        D = (r**2 + zr**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        D = clamp(D, -1.0, 1.0)

        # Elbow-down solution
        q3 = np.arctan2(-np.sqrt(1 - D**2), D)
        q2 = np.arctan2(zr, r) - np.arctan2(
            self.l3 * np.sin(q3),
            self.l2 + self.l3 * np.cos(q3)
        )

        # Clip to joint limits
        q1 = clamp(q1, *self.q_limits[0])
        q2 = clamp(q2, *self.q_limits[1])
        q3 = clamp(q3, *self.q_limits[2])

        return float(q1), float(q2), float(q3)

    # ── JACOBIAN ────────────────────────────────────────────────────────────
    def jacobian(self, q1: float, q2: float, q3: float) -> np.ndarray:
        """
        Analytic 3×3 Jacobian J such that ẋ = J · q̇.
        Rows: [dx/dq, dy/dq, dz/dq] for each joint.
        """
        c1, s1 = np.cos(q1), np.sin(q1)
        c23 = np.cos(q2 + q3)
        s23 = np.sin(q2 + q3)
        r = self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3)
        sh = self.l2 * np.sin(q2) + self.l3 * s23

        J = np.array([
            [-r * s1,   -sh * c1,   -self.l3 * s23 * c1],
            [ r * c1,   -sh * s1,   -self.l3 * s23 * s1],
            [     0.0,   self.l2 * np.cos(q2) + self.l3 * c23,  self.l3 * c23],
        ])
        return J

    def det_jacobian(self, q1: float, q2: float, q3: float) -> float:
        """Determinant of Jacobian — near zero means singularity."""
        return float(np.linalg.det(self.jacobian(q1, q2, q3)))

    def is_singular(self, q1: float, q2: float, q3: float,
                    threshold: float = 1e-3) -> bool:
        return abs(self.det_jacobian(q1, q2, q3)) < threshold

    # ── FORCE → TORQUE ──────────────────────────────────────────────────────
    def force_to_torque(self, q1: float, q2: float, q3: float,
                        f: np.ndarray) -> np.ndarray:
        """
        τ = Jᵀ · f
        f: [fx, fy, fz] end-effector force vector (N)
        Returns: [τ1, τ2, τ3] joint torques (N·m)
        """
        J = self.jacobian(q1, q2, q3)
        return J.T @ f

    # ── CARTESIAN TRAJECTORY ─────────────────────────────────────────────────
    def cartesian_trajectory(self, p_start: np.ndarray, p_end: np.ndarray,
                             n_steps: int = 50):
        """
        Linear interpolation in Cartesian space.
        Returns list of (q1, q2, q3) waypoints.
        Warns if singularity detected along path.
        """
        waypoints = []
        for i, t in enumerate(np.linspace(0, 1, n_steps)):
            p = p_start + t * (p_end - p_start)
            q1, q2, q3 = self.inverse_kinematics(*p)
            if self.is_singular(q1, q2, q3):
                print(f"[ARM WARNING] Singularity near waypoint {i}/{n_steps}, |det J| < 1e-3")
            waypoints.append((q1, q2, q3))
        return waypoints


class PuzzleBotArm:
    """
    Arm entity — wraps PuzzleBotArmModel with runtime joint state.
    """
    def __init__(self, model: PuzzleBotArmModel = None,
                 q_home=(0.0, np.pi/6, -np.pi/4)):
        self.model = model or PuzzleBotArmModel()
        self.q = list(q_home)   # [q1, q2, q3] current joint angles
        self.q_home = list(q_home)
        self.q_target = list(q_home)
        self.q_dot_max = 1.0    # max joint velocity rad/s

    def set_q_target(self, q_target):
        self.q_target = list(q_target)

    def set_ee_target(self, x: float, y: float, z: float):
        q1, q2, q3 = self.model.inverse_kinematics(x, y, z)
        self.q_target = [q1, q2, q3]

    def step(self, dt: float):
        """Move joints toward target at max velocity."""
        for i in range(3):
            err = self.q_target[i] - self.q[i]
            step = clamp(err, -self.q_dot_max * dt, self.q_dot_max * dt)
            self.q[i] += step

    def ee_position(self) -> np.ndarray:
        return self.model.forward_kinematics(*self.q)

    def jacobian(self) -> np.ndarray:
        return self.model.jacobian(*self.q)

    def force_to_torque(self, f: np.ndarray) -> np.ndarray:
        return self.model.force_to_torque(*self.q, f)

    def home(self):
        self.q_target = list(self.q_home)
