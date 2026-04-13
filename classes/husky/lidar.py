import numpy as np
class LiDAR:
    """
    LiDAR 2D simulado, diseñado para montarse sobre un robot móvil.

    Se instancia independientemente y se monta en el robot via
    robot.attach_lidar(lidar), lo que permite swappear sensores
    sin tocar la clase del robot.
    """

    def __init__(
        self,
        n_rays     = 360,
        max_range  = 10.0,
        fov        = 2 * np.pi,
        noise_std  = 0.02,
        mount_offset = (0.0, 0.0),   # offset (dx, dy) relativo al centro del robot
    ):
        self.n_rays       = n_rays
        self.max_range    = max_range
        self.fov          = fov
        self.noise_std    = noise_std
        self.mount_offset = np.array(mount_offset)
        self.angles       = np.linspace(-fov / 2, fov / 2, n_rays)

        # El LiDAR no sabe nada del robot hasta que se monta
        self._robot = None

    def attach(self, robot):
        """Monta el LiDAR en un robot. El robot debe tener get_pose()."""
        self._robot = robot

    def _get_sensor_pose(self):
        """
        Calcula la pose del sensor en el mundo, aplicando el offset de montaje.
        """
        if self._robot is None:
            raise RuntimeError("LiDAR no está montado en ningún robot. Llama attach() primero.")

        x, y, theta = self._robot.get_pose()
        sx = x + self.mount_offset[0] * np.cos(theta) - self.mount_offset[1] * np.sin(theta)
        sy = y + self.mount_offset[0] * np.sin(theta) + self.mount_offset[1] * np.cos(theta)
        return np.array([sx, sy, theta])

    def scan(self, obstacles):
        """
        Escaneo completo desde la pose actual del robot.

        Parameters
        ----------
        obstacles : list de dicts {'x', 'y', 'w', 'h'}

        Returns
        -------
        ranges  : np.ndarray (n_rays,)  — distancias medidas (m)
        angles  : np.ndarray (n_rays,)  — ángulos globales (rad)
        """
        sx, sy, stheta = self._get_sensor_pose()
        global_angles  = stheta + self.angles
        ranges         = np.array([
            self._cast_ray(sx, sy, a, obstacles) for a in global_angles
        ])
        ranges += np.random.normal(0, self.noise_std, self.n_rays)
        return np.clip(ranges, 0.0, self.max_range), global_angles

    def get_detected_boxes(self, obstacles, dist_threshold=0.3):
        """
        Devuelve los centroides estimados de los objetos detectados.
        """
        ranges, angles   = self.scan(obstacles)
        sx, sy, _        = self._get_sensor_pose()

        hits = [
            np.array([sx + r * np.cos(a), sy + r * np.sin(a)])
            for r, a in zip(ranges, angles)
            if r < self.max_range - 0.1
        ]

        return self._cluster_hits(hits, dist_threshold) if hits else []

    # ------------------------------------------------------------------
    # Helpers privados — sin cambios respecto a la versión anterior
    # ------------------------------------------------------------------

    def _cast_ray(self, rx, ry, angle, obstacles):
        dx, dy   = np.cos(angle), np.sin(angle)
        min_dist = self.max_range
        for obs in obstacles:
            d = self._ray_aabb(rx, ry, dx, dy, obs)
            if d is not None and d < min_dist:
                min_dist = d
        return min_dist

    def _ray_aabb(self, rx, ry, dx, dy, box):
        x_min, x_max = box['x'], box['x'] + box['w']
        y_min, y_max = box['y'], box['y'] + box['h']
        t_x = sorted([(x_min - rx) / dx, (x_max - rx) / dx]) if abs(dx) > 1e-9 else [-np.inf,  np.inf]
        t_y = sorted([(y_min - ry) / dy, (y_max - ry) / dy]) if abs(dy) > 1e-9 else [-np.inf,  np.inf]
        t_enter = max(t_x[0], t_y[0])
        t_exit  = min(t_x[1], t_y[1])
        if t_enter < t_exit and t_exit > 0:
            return t_enter if t_enter > 0 else t_exit
        return None

    def _cluster_hits(self, hits, threshold):
        """
        Agrupa puntos usando componentes conectadas (Union-Find).
        A diferencia del algoritmo greedy anterior, si A-B y B-C están
        dentro del umbral, A, B y C quedan en el mismo cluster aunque
        A-C supere el umbral. Esto evita que una caja se fragmente en
        varios centroides.
        """
        n = len(hits)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(hits[i] - hits[j]) < threshold:
                    union(i, j)

        groups: dict = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(hits[i])

        return [np.mean(g, axis=0) for g in groups.values()]