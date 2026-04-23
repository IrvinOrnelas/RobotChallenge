"""
classes/anymal/anymal.py
ANYmal quadruped — per-leg FK/IK/Jacobian + trot gait + singularity monitoring

Convention follows the exam (TE3002B Week 2) exactly:
  q1 : hip abduction/adduction
  q2 : hip flexion/extension (thigh)
  q3 : knee flexion/extension (shank)
  s  : +1 for left legs, -1 for right legs

FK equations (exam-aligned):
  x =  l1*sin(q2) + l2*sin(q2+q3)
  y =  s * l0 * cos(q1)
  z = -l1*cos(q2) - l2*cos(q2+q3)

Jacobian (exam-aligned):
  J = [ [0,          l1*cos(q2)+l2*cos(q2+q3),  l2*cos(q2+q3) ],
        [-s*l0*sin(q1),  0,                      0             ],
        [0,          l1*sin(q2)+l2*sin(q2+q3),  l2*sin(q2+q3) ] ]
"""
import numpy as np
import matplotlib
import matplotlib.patches as patches
from utils import wrap_angle, clamp, norm2


class AnymalLegModel:
    """
    Single leg of ANYmal — kinematics matching the exam convention exactly.

    Parameters
    ----------
    l0  : hip abduction link length (m)   [exam: l0 = 0.0585]
    l1  : thigh link length (m)           [exam: l1 = 0.35]
    l2  : shank link length (m)           [exam: l2 = 0.33]
    s   : side sign (+1 left, -1 right)
    """
    def __init__(self, l0: float = 0.0585, l1: float = 0.35,
                 l2: float = 0.33, s: float = 1.0):
        self.l0 = l0   # hip offset  (was l_hip)
        self.l1 = l1   # thigh
        self.l2 = l2   # shank
        self.s  = s    # side sign

    # ── FORWARD KINEMATICS ──────────────────────────────────────────────────
    def forward_kinematics(self, q1: float, q2: float, q3: float) -> np.ndarray:
        """
        Joint angles → foot position in the leg's shoulder frame.

        Matches exam Q4 equations exactly:
          x =  l1*sin(q2) + l2*sin(q2+q3)
          y =  s * l0 * cos(q1)
          z = -l1*cos(q2) - l2*cos(q2+q3)

        Parameters
        ----------
        q1 : hip abduction angle (rad)
        q2 : hip flexion angle   (rad)
        q3 : knee angle          (rad)

        Returns
        -------
        np.ndarray [x, y, z] foot position (m)
        """
        x = self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)
        y = self.s  * self.l0 * np.cos(q1)
        z = -self.l1 * np.cos(q2) - self.l2 * np.cos(q2 + q3)
        return np.array([x, y, z])

    # ── INVERSE KINEMATICS ──────────────────────────────────────────────────
    def inverse_kinematics(self, px: float, py: float, pz: float):
        """
        Foot position → joint angles (geometric closed-form).

        Parameters
        ----------
        px, py, pz : desired foot position in shoulder frame (m)

        Returns
        -------
        (q1, q2, q3) joint angles in radians
        """
        # Hip abduction: from y = s*l0*cos(q1)
        q1 = np.arccos(clamp(py / (self.s * self.l0), -1.0, 1.0)) if self.l0 > 0 else 0.0

        # In the sagittal plane (x-z):
        # x =  l1*sin(q2) + l2*sin(q2+q3)
        # z = -l1*cos(q2) - l2*cos(q2+q3)
        # → same as a 2-link planar arm with r=sqrt(x²+z²)
        r2 = px**2 + pz**2
        D = clamp((r2 - self.l1**2 - self.l2**2) /
                  (2.0 * self.l1 * self.l2), -1.0, 1.0)

        # Elbow-down solution (knee bends backward)
        q3 = np.arctan2(-np.sqrt(max(0.0, 1.0 - D**2)), D)

        # q2: angle of foot vector from -z axis
        alpha = np.arctan2(px, -pz)   # angle of target in (x, -z) plane
        beta  = np.arctan2(self.l2 * np.sin(q3),
                           self.l1 + self.l2 * np.cos(q3))
        q2 = alpha - beta

        return float(q1), float(q2), float(q3)

    # ── JACOBIAN ────────────────────────────────────────────────────────────
    def jacobian(self, q1: float, q2: float, q3: float) -> np.ndarray:
        """
        3×3 analytic Jacobian — matches exam Q5 exactly.

        J = [ [0,            l1*cos(q2)+l2*cos(q2+q3),  l2*cos(q2+q3) ],
              [-s*l0*sin(q1), 0,                         0             ],
              [0,            l1*sin(q2)+l2*sin(q2+q3),  l2*sin(q2+q3) ] ]

        Columns: ∂p/∂q1, ∂p/∂q2, ∂p/∂q3
        """
        c23 = np.cos(q2 + q3)
        s23 = np.sin(q2 + q3)

        J = np.array([
            [0.0,
             self.l1 * np.cos(q2) + self.l2 * c23,
             self.l2 * c23],
            [-self.s * self.l0 * np.sin(q1),
             0.0,
             0.0],
            [0.0,
             self.l1 * np.sin(q2) + self.l2 * s23,
             self.l2 * s23],
        ])
        return J

    def det_jacobian(self, q1: float, q2: float, q3: float) -> float:
        """
        det(J) in closed form (exam Q5a).
        Expanding along column 1:
          det = -s*l0*sin(q1) * [l1*cos(q2)*l2*sin(q2+q3) - l2*cos(q2+q3)*l1*sin(q2)]
              = -s*l0*sin(q1) * l1*l2 * sin(q3)
        → singular when q1=0,π (hip straight) OR q3=0 (knee fully extended)
        """
        return float(np.linalg.det(self.jacobian(q1, q2, q3)))

    def det_jacobian_symbolic(self, q1: float, q2: float, q3: float) -> float:
        """Closed-form determinant: -s·l0·sin(q1)·l1·l2·sin(q3)"""
        return float(-self.s * self.l0 * np.sin(q1) * self.l1 * self.l2 * np.sin(q3))

    def is_singular(self, q1: float, q2: float, q3: float,
                    threshold: float = 1e-3) -> bool:
        return abs(self.det_jacobian(q1, q2, q3)) < threshold

    def force_to_torque(self, q1: float, q2: float, q3: float,
                        f: np.ndarray) -> np.ndarray:
        """τ = Jᵀ · f  (exam Q6)"""
        return self.jacobian(q1, q2, q3).T @ f

    def det_jacobian(self, q0, q1, q2) -> float:
        return float(np.linalg.det(self.jacobian(q0, q1, q2)))

    def is_singular(self, q0, q1, q2, threshold=1e-3) -> bool:
        return abs(self.det_jacobian(q0, q1, q2)) < threshold


class AnymalLeg:
    """
    Leg entity with current joint state.
    Uses exam-aligned AnymalLegModel (l0, l1, l2, s convention).
    """
    # Shoulder attachment offsets in body frame [x_fwd, y_lat, z_up]
    SHOULDER_OFFSETS = {
        'FL': np.array([ 0.35,  0.20, 0.0]),
        'FR': np.array([ 0.35, -0.20, 0.0]),
        'RL': np.array([-0.35,  0.20, 0.0]),
        'RR': np.array([-0.35, -0.20, 0.0]),
    }
    # Side sign: left=+1, right=-1
    SIDE_SIGN = {'FL': +1.0, 'FR': -1.0, 'RL': +1.0, 'RR': -1.0}

    def __init__(self, name: str, l0=0.0585, l1=0.35, l2=0.33):
        self.name = name
        self.shoulder = self.SHOULDER_OFFSETS[name]
        s = self.SIDE_SIGN[name]
        self.model = AnymalLegModel(l0=l0, l1=l1, l2=l2, s=s)
        # Home config: slight flex, no abduction
        self.q = np.array([0.0, 0.7, -1.4])

    def fk(self) -> np.ndarray:
        """Foot position in shoulder frame."""
        return self.model.forward_kinematics(*self.q)

    def fk_world(self, body_pose: np.ndarray) -> np.ndarray:
        """Foot position in world frame given body pose [x, y, theta]."""
        foot_local = self.fk()
        bx, by, bth = body_pose
        # Rotate shoulder offset by body heading
        so = self.shoulder
        sx = bx + so[0]*np.cos(bth) - so[1]*np.sin(bth) + foot_local[0]*np.cos(bth)
        sy = by + so[0]*np.sin(bth) + so[1]*np.cos(bth) + foot_local[0]*np.sin(bth)
        return np.array([sx, sy, foot_local[2]])

    def set_foot_target(self, px: float, py: float, pz: float):
        """Set joint angles to reach foot position in shoulder frame."""
        q1, q2, q3 = self.model.inverse_kinematics(px, py, pz)
        self.q = np.array([q1, q2, q3])

    def jacobian(self) -> np.ndarray:
        return self.model.jacobian(*self.q)

    def det_J(self) -> float:
        return self.model.det_jacobian(*self.q)

    def check_singularity(self, threshold: float = 1e-3) -> bool:
        """Returns True and prints warning if near singularity."""
        d = abs(self.det_J())
        if d < threshold:
            print(f"[ANYMAL WARNING] Leg {self.name} |det J|={d:.6f} < {threshold} — SINGULAR")
            return True
        return False

    def trot_step(self, phase: float, stride: float = 0.08, lift: float = 0.05):
        """
        Update joint angles for one trot gait step.
        phase : current gait phase (rad)
        stride: peak fore-aft displacement (m)
        lift  : peak foot lift height (m)
        """
        # Nominal foot position in shoulder frame
        nom_x = 0.0
        nom_y = self.model.s * self.model.l0   # abduction rest
        nom_z = -(self.model.l1 + self.model.l2) * 0.75  # ~75% extension

        # Swing: lift and stride
        swing = max(0.0, np.sin(phase))
        target_x = nom_x + stride * np.sin(phase)
        target_y = nom_y
        target_z = nom_z + lift * swing

        self.set_foot_target(target_x, target_y, target_z)
        self.check_singularity()
        
    def setup_visuals(self, ax):
        """Setup Matplotlib artists for a single leg."""
        self.leg_line, = ax.plot([], [], '-o', color='#a78bfa', lw=1.5, ms=3, zorder=5)
        return [self.leg_line]

    def update_visuals(self, body_pose):
        """Update leg line from the hip to the foot's world position."""
        bx, by, bth = body_pose
        so = self.shoulder
        
        # Calculate world position of the hip/shoulder joint
        hip_x = bx + so[0]*np.cos(bth) - so[1]*np.sin(bth)
        hip_y = by + so[0]*np.sin(bth) + so[1]*np.cos(bth)

        # Calculate world position of the foot using your kinematic method
        foot_w = self.fk_world(body_pose)

        self.leg_line.set_data([hip_x, foot_w[0]], [hip_y, foot_w[1]])


class Anymal:
    """
    ANYmal quadruped robot entity.
    Trot gait with 4 legs, payload simulation, singularity monitoring.
    """
    # Diagonal trot pairs: FL+RR in phase, FR+RL offset by π
    TROT_PHASE_OFFSET = {'FL': 0.0, 'RR': 0.0, 'FR': np.pi, 'RL': np.pi}

    def __init__(self, pose=(0.0, 0.0, 0.0), payload_kg: float = 0.0,
                 l0=0.0585, l1=0.35, l2=0.33):
        self.pose = np.array(pose, dtype=float)   # [x, y, theta]
        self.payload_kg = payload_kg              # ~6 kg from 3 PuzzleBots
        self.legs = {
            name: AnymalLeg(name, l0=l0, l1=l1, l2=l2)
            for name in ('FL', 'FR', 'RL', 'RR')
        }
        self.gait_phase = 0.0
        self.trail = []
        self.state = 'IDLE'
        self.v = 0.0
        self.w = 0.0
        self.det_J_log = []   # running log of |det J| per step

    def step(self, dt: float, v: float, w: float):
        """Advance robot one timestep at commanded (v, w)."""
        self.v = v
        self.w = w

        # Integrate pose (Euler)
        self.pose[0] += v * np.cos(self.pose[2]) * dt
        self.pose[1] += v * np.sin(self.pose[2]) * dt
        self.pose[2]  = wrap_angle(self.pose[2] + w * dt)

        # Gait phase advances with speed
        self.gait_phase += v * 5.0 * dt

        # Payload reduces stride
        stride = 0.08 * max(0.2, 1.0 - 0.015 * self.payload_kg)

        dets = {}
        for name, leg in self.legs.items():
            phase = self.gait_phase + self.TROT_PHASE_OFFSET[name]
            leg.trot_step(phase, stride=stride)
            dets[name] = abs(leg.det_J())

        self.det_J_log.append(dets)
        if len(self.det_J_log) > 1000:
            self.det_J_log.pop(0)

        self.trail.append(self.pose[:2].copy())
        if len(self.trail) > 500:
            self.trail.pop(0)

    def navigate_to(self, goal, dt: float, kv: float = 0.4, kw: float = 1.5):
        """
        Proportional controller toward goal [x, y].
        Returns True when within 0.15 m (exam success criterion).
        """
        dx = goal[0] - self.pose[0]
        dy = goal[1] - self.pose[1]
        dist = norm2(dx, dy)
        if dist < 0.15:
            return True
        ang_err = wrap_angle(np.arctan2(dy, dx) - self.pose[2])
        v = clamp(kv * dist, 0.0, 0.8)
        w = clamp(kw * ang_err, -2.0, 2.0)
        self.step(dt, v, w)
        return False

    @property
    def x(self): return self.pose[0]
    @property
    def y(self): return self.pose[1]
    @property
    def theta(self): return self.pose[2]

    def all_jacobians(self) -> dict:
        return {name: leg.jacobian() for name, leg in self.legs.items()}

    def check_all_singularities(self, threshold: float = 1e-3) -> dict:
        """Check all legs for singularity. Returns dict name→bool."""
        return {name: leg.check_singularity(threshold)
                for name, leg in self.legs.items()}

    def min_det_J(self) -> float:
        """Minimum |det J| across all legs (safety metric)."""
        return min(abs(leg.det_J()) for leg in self.legs.values())
    
    def setup_visuals(self, ax):
        """Setup Matplotlib artists for the ANYmal body and legs."""
        # Trail
        self.trail_line, = ax.plot([], [], '-', color='#7c3aed', lw=0.8, alpha=0.4, zorder=2)
        
        # Body
        self.body_patch = patches.Ellipse((0, 0), 0.7, 0.4, fc='#7c3aed', ec='#c4b5fd', lw=1.5, zorder=5)
        ax.add_patch(self.body_patch)
        
        # Label
        self.label = ax.text(0, 0, 'ANYmal', color='#c4b5fd', 
                             fontsize=6, fontfamily='monospace', zorder=6)

        artists = [self.trail_line, self.body_patch, self.label]
        for leg in self.legs.values():
            artists.extend(leg.setup_visuals(ax))
        return artists

    def update_visuals(self, ax):
        """Update visual state per frame."""
        if len(self.trail) > 1:
            tr = np.array(self.trail)
            self.trail_line.set_data(tr[:,0], tr[:,1])

        # Apply body rotation and translation
        t_mat = matplotlib.transforms.Affine2D().rotate(self.theta).translate(self.x, self.y)
        self.body_patch.set_transform(t_mat + ax.transData)

        self.label.set_position((self.x + 0.42, self.y + 0.25))

        # Update all legs
        for leg in self.legs.values():
            leg.update_visuals(self.pose)