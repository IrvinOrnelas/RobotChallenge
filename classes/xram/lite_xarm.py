import numpy as np
import matplotlib.patches as patches

class Lite6ArmModel3D:
    def __init__(self):
        # Approximate DH parameters for Lite6
        # [theta_offset, d, a, alpha]
        self.dh = [
            [0,    0.243, 0.0,   np.pi/2],
            [0,    0.0,   0.200, 0],
            [0,    0.0,   0.087, np.pi/2],
            [0,    0.227, 0.0,  -np.pi/2],
            [0,    0.0,   0.0,   np.pi/2],
            [0,    0.0615,0.0,   0]
        ]

    # -----------------------------
    # DH Transform
    # -----------------------------
    def dh_transform(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,      sa,     ca,    d],
            [0,       0,      0,    1]
        ])

    # -----------------------------
    # Forward Kinematics (3D)
    # -----------------------------
    def forward_kinematics(self, q):
        T = np.eye(4)
        positions = []
        z_axes = []

        for i in range(6):
            theta = q[i] + self.dh[i][0]
            d, a, alpha = self.dh[i][1:]

            T = T @ self.dh_transform(theta, d, a, alpha)

            positions.append(T[:3, 3])
            z_axes.append(T[:3, 2])  # joint axis

        return T, positions, z_axes

    # -----------------------------
    # Jacobian (Position Only)
    # -----------------------------
    def compute_jacobian(self, q):
        T, positions, z_axes = self.forward_kinematics(q)

        J = np.zeros((3, 6))
        p_end = positions[-1]

        for i in range(6):
            zi = z_axes[i]
            pi = positions[i] if i == 0 else positions[i-1]

            J[:, i] = np.cross(zi, (p_end - pi))

        return J

    # -----------------------------
    # Numerical IK (Jacobian pseudo-inverse)
    # -----------------------------
    def inverse_kinematics(self, target_pos, q_init=None, max_iter=200):
        if q_init is None:
            q = np.zeros(6)
        else:
            q = q_init.copy()

        for _ in range(max_iter):
            T, _, _ = self.forward_kinematics(q)
            current_pos = T[:3, 3]

            error = target_pos - current_pos

            if np.linalg.norm(error) < 1e-3:
                break

            J = self.compute_jacobian(q)

            # Damped least squares (more stable than pinv)
            lambda_ = 0.1
            JT = J.T
            dq = JT @ np.linalg.inv(J @ JT + lambda_**2 * np.eye(3)) @ error

            q += dq

        return q

    # -----------------------------
    # 2D Projection (Top View)
    # -----------------------------
    def project_to_2d(self, positions):
        projected = [(p[0], p[1]) for p in positions]
        return projected


# ============================================================================
# SIMULATION ENTITY WRAPPER
# ============================================================================

class Lite6Arm:
    """
    Wraps the 6-DOF mathematical model for the 2D Simulator.
    Manages world position, joint interpolation, and Matplotlib rendering.
    """
    def __init__(self, arm_id, x, y):
        self.id = arm_id
        self.x = x
        self.y = y
        
        self.model = Lite6ArmModel3D()
        
        # State Arrays
        self.q = np.array([0.0, 0.5, -1.0, 0.0, 0.5, 0.0]) # A nice resting pose
        self.target_q = self.q.copy()
        self.q_speed = 2.0  # Joint speed in rad/s
        
        # Tracking end-effector in world space
        self.ee_x = x
        self.ee_y = y
        self.joint_positions_2d = []

    def set_target(self, tx, ty, tz=0.0):
        """Requests the IK solver to find joint angles for a world coordinate."""
        # Convert world coordinates (tx, ty) to robot's local base frame
        local_target = np.array([tx - self.x, ty - self.y, tz])
        
        # Use your Numerical IK solver
        self.target_q = self.model.inverse_kinematics(local_target, q_init=self.q)

    def step(self, dt):
        """Moves joints toward target_q at q_speed. Updates FK projections."""
        arrived = True
        for i in range(6):
            err = self.target_q[i] - self.q[i]
            if abs(err) > 0.05:
                arrived = False
                step = np.sign(err) * min(abs(err), self.q_speed * dt)
                self.q[i] += step
                
        # Compute 3D Forward Kinematics for the current interpolated state
        _, positions, _ = self.model.forward_kinematics(self.q)
        
        # Add the base origin (0,0,0) to the beginning of the chain
        all_positions = [np.array([0, 0, 0])] + positions
        
        # Project down to 2D
        proj_2d = self.model.project_to_2d(all_positions)
        
        # Shift local 2D projections to Global World Coordinates
        self.joint_positions_2d = [(px + self.x, py + self.y) for (px, py) in proj_2d]
        
        # Update public end-effector property for the Coordinator to use
        self.ee_x = self.joint_positions_2d[-1][0]
        self.ee_y = self.joint_positions_2d[-1][1]
        
        return arrived

    def setup_visuals(self, ax):
        # Draw the base
        self.base_patch = patches.Circle((self.x, self.y), 0.15, fc='#1e3a8a', ec='#60a5fa', lw=1.5, zorder=3)
        ax.add_patch(self.base_patch)
        
        # Draw a segmented line for all 6 links with dots for joints!
        self.arm_line, = ax.plot([], [], '-o', color='#3b82f6', mfc='#60a5fa', mec='white', lw=3, ms=5, zorder=6)
        
        # Label
        self.label = ax.text(self.x, self.y + 0.25, f'Lite6_{self.id}', color='#93c5fd', 
                             fontsize=6, ha='center', fontfamily='monospace', zorder=7)
                             
        return [self.base_patch, self.arm_line, self.label]

    def update_visuals(self, ax=None):
        if len(self.joint_positions_2d) > 0:
            xs = [p[0] for p in self.joint_positions_2d]
            ys = [p[1] for p in self.joint_positions_2d]
            self.arm_line.set_data(xs, ys)