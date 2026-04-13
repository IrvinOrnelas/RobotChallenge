from classes.puzzlebot.puzzlebot_arm import PuzzleBotArm, PuzzleBotArmModel
import numpy as np  

class PuzzleBotModel:
    def __init__(self, r = 0.05, L = 0.19, m = 2):
        self.r = r
        self.L = L
        self.m = m
        self.max_wheel_speed = 0.8

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def forward_kinematics(self, omega  = None):
        if omega is None:
            return None
        
        wr, wl = omega

        v = (self.r / 2) * (wr + wl)
        w = (self.r / self.L) * (wr - wl)

        return np.array([v, w])
    
    def inverse_kinematics(self, states):
        if states is None:
            return None

        v, w = states

        wr = (2*v + w*self.L)/(2*self.r)
        wl = (2*v - w*self.L)/(2*self.r)

        return np.array([wr,wl])
    
    def integrate(self, pose, omega, dt):
        x, y, theta = pose

        v, w = self.forward_kinematics(omega)

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        theta = self.wrap_to_pi(theta)

        return np.array([x, y, theta])

    def jacobian(self, theta, p_ee_base):
        x, y, _ = p_ee_base

        J = np.array([
            [np.cos(theta), -y],
            [np.sin(theta),  x],
            [0,              0]
        ])

        return J
    
    def clamp_wheels(self, omega):
        return np.clip(omega, -self.max_wheel_speed, self.max_wheel_speed)

class PuzzleBot:
    def __init__(self, model: PuzzleBotModel, arm: PuzzleBotArm):
        self.model = model
        self.arm = arm

        # Estado base: [x, y, theta]
        self.pose = np.array([0.0, 0.0, 0.0])

        # Velocidades actuales
        self.omega = np.array([0.0, 0.0])  # [wr, wl]

    # -------------------------------------------------
    # Control
    # -------------------------------------------------

    def set_wheel_speeds(self, omega):
        omega = np.array(omega)
        self.omega = self.model.clamp_wheels(omega)

    def set_twist(self, v, w):
        omega = self.model.inverse_kinematics([v, w])
        self.set_wheel_speeds(omega)

    # -------------------------------------------------
    # Simulación (step)
    # -------------------------------------------------

    def step(self, dt):
        """
        Avanza el estado del robot
        """
        self.pose = self.model.integrate(self.pose, self.omega, dt)

    # -------------------------------------------------
    # Transformaciones
    # -------------------------------------------------

    def base_to_world(self, p_base):
        x, y, theta = self.pose

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        t = np.array([x, y, 0])

        return R @ p_base + t

    def get_ee_world(self):
        """
        End-effector en coordenadas globales
        """
        p_arm = self.arm.p
        return self.base_to_world(p_arm)

    # -------------------------------------------------
    # Debug / estado
    # -------------------------------------------------

    def get_state(self):
        return {
            "pose": self.pose.tolist(),
            "omega": self.omega.tolist(),
            "ee_world": self.get_ee_world().tolist()
        }

    def print_state(self):
        s = self.get_state()
        print("=" * 40)
        print("PuzzleBot State")
        print("=" * 40)
        print(f"Pose     : {s['pose']}")
        print(f"Omega    : {s['omega']}")
        print(f"EE world : {s['ee_world']}")
