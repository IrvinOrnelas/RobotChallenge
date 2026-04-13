import numpy as np
import time

class PuzzleBotArmModel:
    def __init__(self, l1=0.10, l2=0.08, l3=0.06):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def forward_kinematics(self, q=None):
        if q is None:
            return None

        q1, q2, q3 = q

        s1 = np.sin(q1)
        c1 = np.cos(q1)

        s2  = np.sin(q2)
        c2  = np.cos(q2)

        s23 = np.sin(q2 + q3)
        c23 = np.cos(q2 + q3)

        r = self.l2 * c2 + self.l3 * c23

        x = r * c1
        y = r * s1
        z = self.l1 + self.l2 * s2 + self.l3 * s23

        return np.array([x, y, z])

    def inverse_kinematics(self, p=None):
        if p is None:
            return None

        x, y, z = p

        r  = np.sqrt(x**2 + y**2)
        z_ = z - self.l1

        c3 = (r**2 + z_**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        c3 = np.clip(c3, -1.0, 1.0)
        s3 = np.sqrt(1 - c3**2)

        q3 = np.arctan2(s3, c3)
        q2 = np.arctan2(z_, r) - np.arctan2(self.l3 * s3, self.l2 + self.l3 * c3)
        q1 = np.arctan2(y, x)

        return np.array([q1, q2, q3])

    def jacobian(self, q=None):
        q1, q2, q3 = q

        s1 = np.sin(q1)
        c1 = np.cos(q1)

        s2  = np.sin(q2)
        c2  = np.cos(q2)

        s23 = np.sin(q2 + q3)
        c23 = np.cos(q2 + q3)

        J = np.array([
            [-s1*(self.l2*c2 + self.l3*c23),  c1*(-self.l2*s2 - self.l3*s23), -self.l3*s23*c1],
            [ c1*(self.l2*c2 + self.l3*c23),  s1*(-self.l2*s2 - self.l3*s23), -self.l3*s23*s1],
            [ 0,                               self.l2*c2 + self.l3*c23,        self.l3*c23   ]
        ])
        return J

    def force_to_torque(self, f_tip, q=None):
        J = self.jacobian(q)
        if J is None:
            return None
        return J.T @ f_tip

    def check_singularity(self, q, threshold=1e-5):
        J   = self.jacobian(q)
        det = np.linalg.det(J)
        return abs(det) < threshold, abs(det)

    def cartesian_trajectory(self, p_start, p_end, n_steps=50):
        qs = []
        for i in range(n_steps + 1):
            t = i / n_steps
            p_interp = (1 - t) * p_start + t * p_end

            q = self.inverse_kinematics(p_interp)
            if q is None:
                print(f"[WARN] IK fallo en paso {i}, punto {p_interp}")
                break

            singular, det = self.check_singularity(q)
            if singular:
                print(f"[WARN] Singularidad detectada en paso {i}, det={det:.2e}")

            qs.append(q)
        return qs

    def check_fk_ik_consistency(self, q_test=None):
        if q_test is None:
            q_test = np.array([0.3, 0.4, -0.6])

        p_original  = self.forward_kinematics(q_test)
        q_recovered = self.inverse_kinematics(p_original)
        p_recovered = self.forward_kinematics(q_recovered)

        error = np.linalg.norm(p_original - p_recovered)
        print(f"Error FK->IK->FK: {error:.2e}")

        return error < 1e-6

    def differential_ik(self, q, v_cart):
        J = self.jacobian(q)

        J_pinv = np.linalg.pinv(J)

        q_dot = J_pinv @ v_cart
        return q_dot

# ---------------------------------------------------------------------------
# PuzzleBotArm 
# ---------------------------------------------------------------------------

class PuzzleBotArm:
    JOINT_NAMES = ["q1 (base)", "q2 (hombro)", "q3 (codo)"]

    def __init__(
        self,
        model: PuzzleBotArmModel,
        q_home=None,
        joint_limits=None,
        sim_delay: float = 0.0,
    ):
        self.model = model
        self.sim_delay = sim_delay

        # Posición articular home
        self.q_home = np.array(q_home, dtype=float) if q_home is not None \
                      else np.zeros(3)

        # Límites articulares: lista de tuplas (q_min, q_max) por junta
        if joint_limits is not None:
            self.joint_limits = [tuple(lim) for lim in joint_limits]
        else:
            self.joint_limits = [(-np.pi, np.pi)] * 3

        # Estado interno
        self._q: np.ndarray = self.q_home.copy()   # posición articular actual
        self._p: np.ndarray | None = self.model.forward_kinematics(self._q)

        # Almacén de waypoints nombrados  {nombre: q_array}
        self._waypoints: dict[str, np.ndarray] = {}

        # Historial: lista de dicts con info de cada movimiento ejecutado
        self._history: list[dict] = []

        # Flag de estado
        self._is_enabled: bool = False

    # ------------------------------------------------------------------
    # Propiedades de solo lectura
    # ------------------------------------------------------------------

    @property
    def q(self) -> np.ndarray:
        """Posición articular actual [q1, q2, q3] en radianes."""
        return self._q.copy()

    @property
    def p(self) -> np.ndarray | None:
        """Posición cartesiana actual [x, y, z] en metros."""
        return self._p.copy() if self._p is not None else None

    @property
    def is_enabled(self) -> bool:
        """True si el brazo está habilitado para recibir comandos."""
        return self._is_enabled

    @property
    def history(self) -> list[dict]:
        """Historial de movimientos (solo lectura)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Habilitación / deshabilitación
    # ------------------------------------------------------------------

    def enable(self):
        """Habilita el brazo (simula activación de motores)."""
        self._is_enabled = True
        print("[ARM] Brazo habilitado.")

    def disable(self):
        """Deshabilita el brazo (simula desactivación de motores)."""
        self._is_enabled = False
        print("[ARM] Brazo deshabilitado.")

    # ------------------------------------------------------------------
    # Límites articulares
    # ------------------------------------------------------------------

    def _check_limits(self, q: np.ndarray) -> bool:
        """
        Verifica que q esté dentro de los límites articulares.

        Retorna True si es válido, False si alguna junta está fuera de rango.
        """
        for i, (qi, (q_min, q_max)) in enumerate(zip(q, self.joint_limits)):
            if not (q_min <= qi <= q_max):
                print(
                    f"[ARM] ⚠  {self.JOINT_NAMES[i]} = {np.degrees(qi):.2f}° "
                    f"fuera del rango [{np.degrees(q_min):.1f}°, {np.degrees(q_max):.1f}°]"
                )
                return False
        return True

    def _clamp_limits(self, q: np.ndarray) -> np.ndarray:
        """Aplica saturación a q respetando los límites articulares."""
        q_clamped = q.copy()
        for i, (q_min, q_max) in enumerate(self.joint_limits):
            q_clamped[i] = np.clip(q[i], q_min, q_max)
        return q_clamped

    # ------------------------------------------------------------------
    # Envío de comandos al hardware (stub)
    # ------------------------------------------------------------------

    def send_joint_command(self, q: np.ndarray):
        """
        Envía la configuración articular al hardware real.

        En la implementación real este método se comunica con el
        microcontrolador / driver de servos. Aquí es un stub que imprime
        los valores y aplica el retardo de simulación.
        """
        deg = np.degrees(q)
        print(
            f"[HW]  → q = [{deg[0]:+7.2f}°, {deg[1]:+7.2f}°, {deg[2]:+7.2f}°]"
        )
        if self.sim_delay > 0:
            time.sleep(self.sim_delay)

    # ------------------------------------------------------------------
    # Movimiento articular directo
    # ------------------------------------------------------------------

    def move_to_joints(self, q_target, label: str = "") -> bool:
        """
        Mueve el brazo directamente a la configuración articular q_target.

        Parámetros
        ----------
        q_target : array-like
            Ángulos objetivo [q1, q2, q3] en radianes.
        label : str
            Etiqueta opcional para el historial.

        Retorna
        -------
        bool : True si el movimiento se completó con éxito.
        """
        if not self._is_enabled:
            print("[ARM] ERROR: El brazo no está habilitado.")
            return False

        q_target = np.array(q_target, dtype=float)

        if not self._check_limits(q_target):
            print("[ARM] Movimiento cancelado por límites articulares.")
            return False

        singular, det = self.model.check_singularity(q_target)
        if singular:
            print(f"[ARM] ⚠  Configuración singular (det={det:.2e}), movimiento cancelado.")
            return False

        self.send_joint_command(q_target)

        self._q = q_target.copy()
        self._p = self.model.forward_kinematics(self._q)

        self._history.append({
            "type":  "joint",
            "label": label,
            "q":     self._q.copy(),
            "p":     self._p.copy() if self._p is not None else None,
        })
        return True

    # ------------------------------------------------------------------
    # Movimiento cartesiano (IK puntual)
    # ------------------------------------------------------------------

    def move_to_cartesian(self, p_target, label: str = "") -> bool:
        """
        Mueve el efector final a la posición cartesiana p_target.

        Utiliza la cinemática inversa del modelo. El movimiento NO es
        una línea recta en el espacio cartesiano; para eso usar
        move_cartesian_line().

        Parámetros
        ----------
        p_target : array-like
            Posición objetivo [x, y, z] en metros.
        label : str
            Etiqueta opcional para el historial.

        Retorna
        -------
        bool : True si el movimiento se completó con éxito.
        """
        p_target = np.array(p_target, dtype=float)
        q_target = self.model.inverse_kinematics(p_target)

        if q_target is None:
            print("[ARM] ERROR: IK no encontró solución para el punto objetivo.")
            return False

        return self.move_to_joints(q_target, label=label)

    # ------------------------------------------------------------------
    # Trayectoria cartesiana lineal
    # ------------------------------------------------------------------

    def move_cartesian_line(
        self,
        p_end,
        n_steps: int = 30,
        label: str = "",
        clamp_limits: bool = False,
    ) -> bool:
        """
        Mueve el efector final en línea recta desde la posición actual
        hasta p_end, siguiendo n_steps pasos interpolados.

        Parámetros
        ----------
        p_end : array-like
            Posición cartesiana final [x, y, z] en metros.
        n_steps : int
            Número de puntos intermedios.
        label : str
            Etiqueta para el historial.
        clamp_limits : bool
            Si True, satura las juntas a sus límites en lugar de abortar.

        Retorna
        -------
        bool : True si se completó la trayectoria completa.
        """
        if not self._is_enabled:
            print("[ARM] ERROR: El brazo no está habilitado.")
            return False

        if self._p is None:
            print("[ARM] ERROR: Posición cartesiana actual desconocida.")
            return False

        p_start  = self._p.copy()
        p_end    = np.array(p_end, dtype=float)
        qs       = self.model.cartesian_trajectory(p_start, p_end, n_steps)

        print(f"[ARM] Trayectoria lineal: {len(qs)} pasos.")

        for i, q_step in enumerate(qs):
            if clamp_limits:
                q_step = self._clamp_limits(q_step)
            elif not self._check_limits(q_step):
                print(f"[ARM] Trayectoria abortada en paso {i} por límites articulares.")
                return False

            self.send_joint_command(q_step)

        # Actualizar estado al último punto
        self._q = qs[-1].copy()
        self._p = self.model.forward_kinematics(self._q)

        self._history.append({
            "type":    "cartesian_line",
            "label":   label,
            "p_start": p_start,
            "p_end":   p_end,
            "n_steps": len(qs),
            "q":       self._q.copy(),
            "p":       self._p.copy() if self._p is not None else None,
        })
        return True

    # ------------------------------------------------------------------
    # Home
    # ------------------------------------------------------------------

    def go_home(self) -> bool:
        """Mueve el brazo a la posición articular home."""
        print("[ARM] Regresando a home…")
        return self.move_to_joints(self.q_home, label="home")

    # ------------------------------------------------------------------
    # Waypoints
    # ------------------------------------------------------------------

    def save_waypoint(self, name: str, q: np.ndarray | None = None):
        """
        Guarda una configuración articular con un nombre.

        Parámetros
        ----------
        name : str
            Nombre del waypoint.
        q : array-like, opcional
            Configuración a guardar. Si es None usa la posición actual.
        """
        q_save = np.array(q, dtype=float) if q is not None else self._q.copy()
        self._waypoints[name] = q_save
        print(f"[ARM] Waypoint '{name}' guardado: {np.degrees(q_save).round(2)} °")

    def go_to_waypoint(self, name: str) -> bool:
        """
        Mueve el brazo al waypoint guardado con ese nombre.

        Retorna False si el nombre no existe.
        """
        if name not in self._waypoints:
            print(f"[ARM] ERROR: Waypoint '{name}' no existe.")
            return False
        print(f"[ARM] Moviendo a waypoint '{name}'…")
        return self.move_to_joints(self._waypoints[name], label=name)

    def list_waypoints(self) -> list[str]:
        """Retorna los nombres de los waypoints guardados."""
        return list(self._waypoints.keys())

    def delete_waypoint(self, name: str) -> bool:
        """Elimina un waypoint por nombre. Retorna False si no existe."""
        if name not in self._waypoints:
            print(f"[ARM] ERROR: Waypoint '{name}' no existe.")
            return False
        del self._waypoints[name]
        print(f"[ARM] Waypoint '{name}' eliminado.")
        return True

    # ------------------------------------------------------------------
    # Estado y diagnóstico
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Retorna un dict con el estado completo del brazo.

        Incluye: posición articular, posición cartesiana, estado de
        habilitación, singularidad y límites articulares.
        """
        singular, det = self.model.check_singularity(self._q)
        return {
            "enabled":   self._is_enabled,
            "q_deg":     np.degrees(self._q).round(4).tolist(),
            "q_rad":     self._q.round(6).tolist(),
            "p_m":       self._p.round(6).tolist() if self._p is not None else None,
            "singular":  singular,
            "det_J":     round(det, 6),
            "waypoints": list(self._waypoints.keys()),
        }

    def print_state(self):
        """Imprime el estado actual del brazo de forma legible."""
        s = self.get_state()
        print("=" * 50)
        print("  Estado del PuzzleBotArm")
        print("=" * 50)
        print(f"  Habilitado : {s['enabled']}")
        print(f"  q (grados) : {s['q_deg']}")
        print(f"  q (rad)    : {s['q_rad']}")
        print(f"  p (m)      : {s['p_m']}")
        print(f"  Singular   : {s['singular']}  (det J = {s['det_J']:.4e})")
        print(f"  Waypoints  : {s['waypoints']}")
        print("=" * 50)

    def clear_history(self):
        """Limpia el historial de movimientos."""
        self._history.clear()
        print("[ARM] Historial limpiado.")

    # ------------------------------------------------------------------
    # Representación textual
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        q_deg = np.degrees(self._q).round(2)
        return (
            f"PuzzleBotArm("
            f"enabled={self._is_enabled}, "
            f"q={q_deg}°, "
            f"waypoints={list(self._waypoints.keys())})"
        )


