import numpy as np
from lidar import LiDAR

class HuskyModel:
    def __init__(self, r = 0.1651, B = 0.555, m = 50, maxspeed = 1):
        self.r = r
        self.B = B
        self.m = 50
        self.maxspeed = maxspeed

    def forward_kinematics(self, omega, s):
        if omega is None:
            return None
        
        if s is None:
            s = 1
        
        wr1, wr2, wl1, wl2 = omega

        v = (self.r/4)*(wr1 + wr2 + wl1 + wl2) * s
        w = (self.r/(2*self.B))*(wr1 + wr2 - wl1 - wl2)

        return np.array([v,w])
    
    def integrate(self, pose, omega, s, dt):
        if pose is None or omega is None:
            return None

        x, y, theta = pose

        v,w = self.forward_kinematics(omega, s)
        theta_mid = theta + w * (dt / 2)

        x += v * np.cos(theta_mid) * dt
        y += v * np.sin(theta_mid) * dt
        theta += w * dt

        return np.array([x, y, theta])

    def inverse_kinematics(self, v, w):
        wr = (2*v + w*self.B) / (2*self.r)
        wl = (2*v - w*self.B) / (2*self.r)
        # Clamp a maxspeed
        scale = max(abs(wr), abs(wl)) / self.maxspeed
        if scale > 1:
            wr /= scale
            wl /= scale
        return np.array([wr, wr, wl, wl])
    
    def compute_velocity_command(self, pose, goal, k_v=0.5, k_w=2.0):
        x, y, theta = pose
        dx, dy = goal[0] - x, goal[1] - y
        dist = np.hypot(dx, dy)
        angle_to_goal = np.arctan2(dy, dx)
        angle_error = np.arctan2(
            np.sin(angle_to_goal - theta),
            np.cos(angle_to_goal - theta)
        )
        v = k_v * dist
        w = k_w * angle_error

        return v, w
    
    def get_telemetry(self, omega_cmd, omega_meas, s):
        v_cmd, w_cmd   = self.forward_kinematics(omega_cmd,  s)
        v_meas, w_meas = self.forward_kinematics(omega_meas, s)
        return {
            "v_cmd": v_cmd,   "w_cmd": w_cmd,
            "v_meas": v_meas, "w_meas": w_meas,
            "slip": s
        }

class Husky:
    """
    Instancia física (o simulada) del Husky A200.
    Puede tener un LiDAR montado como componente opcional.
    """

    def __init__(self, pose=(0.0, 0.0, 0.0), s=1.0):
        self.model = HuskyModel()
        self.pose  = np.array(pose, dtype=float)
        self.s     = s

        self._omega_cmd  = np.zeros(4)
        self._omega_meas = np.zeros(4)
        self.log         = []

        # Sensor — None hasta que se monte uno
        self.lidar: LiDAR | None = None

    def attach_lidar(self, lidar: LiDAR):
        """Monta un LiDAR en el robot."""
        lidar.attach(self)
        self.lidar = lidar

    def scan(self, obstacles):
        """
        Dispara el LiDAR si está montado.
        Lanza RuntimeError si no hay sensor.
        """
        if self.lidar is None:
            raise RuntimeError("No hay LiDAR montado. Llama attach_lidar() primero.")
        return self.lidar.scan(obstacles)

    def detect_boxes(self, obstacles):
        """Devuelve centroides de cajas detectadas por el LiDAR."""
        if self.lidar is None:
            raise RuntimeError("No hay LiDAR montado. Llama attach_lidar() primero.")
        return self.lidar.get_detected_boxes(obstacles)

    # ------------------------------------------------------------------
    # El resto sin cambios
    # ------------------------------------------------------------------

    def set_slip(self, s):
        self.s = float(np.clip(s, 0.0, 1.0))

    def send_velocity(self, v, w):
        self._omega_cmd = self.model.inverse_kinematics(v, w)

    def step(self, dt, noise_std=0.02):
        self._omega_meas = self._omega_cmd + np.random.normal(0, noise_std, 4)
        self.pose        = self.model.integrate(self.pose, self._omega_meas, self.s, dt)
        telem            = self.model.get_telemetry(self._omega_cmd, self._omega_meas, self.s)
        telem["pose"]    = self.pose.copy()
        self.log.append(telem)

    def move_to(self, goal, dt=0.05, tol=0.1, max_steps=2000):
        goal = np.array(goal, dtype=float)
        for _ in range(max_steps):
            if np.hypot(*(goal - self.pose[:2])) < tol:
                self.send_velocity(0.0, 0.0)
                return True
            v, w = self.model.compute_velocity_command(self.pose, goal)
            self.send_velocity(v, w)
            self.step(dt)
        return False

    def get_pose(self):
        return self.pose.copy()

    def reset(self, pose=(0.0, 0.0, 0.0)):
        self.pose        = np.array(pose, dtype=float)
        self._omega_cmd  = np.zeros(4)
        self._omega_meas = np.zeros(4)
        self.log.clear()