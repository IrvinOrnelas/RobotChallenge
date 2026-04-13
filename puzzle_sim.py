"""
sim_visual.py — Simulación visual del PuzzleBot completo (Matplotlib + Qt).

Ventana dividida en:
  izquierda  – vista 3-D del brazo (links coloreados + trail del EE)
  derecha    – vista cenital 2-D  (trayectoria de la base + orientación)

Fases animadas:
  0  Pose inicial
  1  Base avanza en línea recta
  2  Brazo → home
  3  Brazo → punto cartesiano (IK)
  4  Brazo ejecuta trayectoria lineal cartesiana
  5  Base gira ~90 °
  6  Brazo vuelve a home
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (activa 3-D)

from classes.puzzlebot.puzzlebot import PuzzleBot, PuzzleBotModel
from classes.puzzlebot.puzzlebot_arm import PuzzleBotArm, PuzzleBotArmModel


# ══════════════════════════════════════════════════════════════════════
# Paleta de colores
# ══════════════════════════════════════════════════════════════════════
BG     = '#0d0d1a'
PANEL  = '#12192e'
GRID   = '#1c2b42'
WHITE  = '#ddeeff'
YELLOW = '#ffd93d'
RED    = '#ff6b6b'
GREEN  = '#6bcb77'
BLUE   = '#4d96ff'
ORANGE = '#ff9d3d'
PURPLE = '#c77dff'

LINK_CLR = [RED, YELLOW, GREEN]   # base-link, shoulder, elbow


# ══════════════════════════════════════════════════════════════════════
# Geometría auxiliar
# ══════════════════════════════════════════════════════════════════════

def arm_joints_world(pose, arm_model, q):
    """
    Calcula las 4 posiciones (juntas) del brazo en coordenadas mundo.

    Retorna array (4, 3):
        [0] = base del brazo (sobre el robot)
        [1] = junta 1  (después de l1 vertical)
        [2] = junta 2  (después de l2 hombro)
        [3] = efector final
    """
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c23    = np.cos(q2 + q3)
    s23    = np.sin(q2 + q3)
    l1, l2, l3 = arm_model.l1, arm_model.l2, arm_model.l3

    # Posiciones en frame del robot (body frame)
    pts_local = np.array([
        [0.0,               0.0,               0.0              ],
        [0.0,               0.0,               l1               ],
        [l2 * c2 * c1,      l2 * c2 * s1,      l1 + l2 * s2    ],
        [(l2*c2 + l3*c23)*c1, (l2*c2 + l3*c23)*s1, l1 + l2*s2 + l3*s23],
    ])

    # Transformación al marco mundo (rotación 2-D en plano XY + traslación)
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0., 0., 1.0]])
    t = np.array([x, y, 0.0])

    return (R @ pts_local.T).T + t


def robot_corners(pose, hw=0.085, hh=0.060):
    """Vértices del rectángulo del cuerpo del robot en XY."""
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s, c]])
    local = np.array([[ hh,  hw],
                      [ hh, -hw],
                      [-hh, -hw],
                      [-hh,  hw]])
    return (R2 @ local.T).T + np.array([x, y])


# ══════════════════════════════════════════════════════════════════════
# Pre-computación de fotogramas
# ══════════════════════════════════════════════════════════════════════

def build_frames():
    """
    Ejecuta la simulación cinemática y devuelve una lista de fotogramas.
    Cada fotograma: dict  {'pose': np.array(3), 'q': np.array(3), 'label': str}
    """
    arm_model = PuzzleBotArmModel(l1=0.10, l2=0.08, l3=0.06)
    arm = PuzzleBotArm(
        model=arm_model,
        q_home=[0.0, np.pi / 6, -np.pi / 4],
        joint_limits=[(-np.pi, np.pi),
                      (-np.pi / 2, np.pi / 2),
                      (-np.pi / 2, np.pi / 2)],
        sim_delay=0.0,
    )
    bot_model = PuzzleBotModel(r=0.05, L=0.19)
    bot       = PuzzleBot(model=bot_model, arm=arm)
    arm.enable()

    DT = 0.05
    frames = []

    def snap(label):
        frames.append({
            'pose':  bot.pose.copy(),
            'q':     arm._q.copy(),
            'label': label,
        })

    # ── Fase 0: pose inicial ─────────────────────────────────────────
    for _ in range(30):
        snap("Inicio")

    # ── Fase 1: avance en línea recta ────────────────────────────────
    # set_twist(0.20, 0) → ruedas requieren 4 rad/s → saturan a 0.8 rad/s
    # v_real = (r/2)·(0.8+0.8) = 0.04 m/s   →  100 pasos × 0.05 s = 0.20 m
    bot.set_twist(0.20, 0.0)
    for _ in range(100):
        bot.step(DT)
        snap("Base: avance recto  →  v_real ≈ 0.04 m/s")

    # ── Fase 2: brazo → home ─────────────────────────────────────────
    q_from = arm._q.copy()
    q_to   = arm.q_home.copy()
    for t in np.linspace(0.0, 1.0, 50):
        arm._q = (1 - t) * q_from + t * q_to
        snap("Brazo → home  [q1=0°  q2=30°  q3=-45°]")
    arm._q = q_to.copy()

    # ── Fase 3: IK puntual hacia objetivo cartesiano ─────────────────
    p_goal = np.array([0.08, 0.04, 0.16])
    q_goal = arm_model.inverse_kinematics(p_goal)
    q_from = arm._q.copy()
    for t in np.linspace(0.0, 1.0, 60):
        arm._q = (1 - t) * q_from + t * q_goal
        snap(f"Brazo → IK  target=({p_goal[0]:.2f},{p_goal[1]:.2f},{p_goal[2]:.2f}) m")
    arm._q = q_goal.copy()

    # breve pausa
    for _ in range(15):
        snap(f"Brazo → IK  target=({p_goal[0]:.2f},{p_goal[1]:.2f},{p_goal[2]:.2f}) m")

    # ── Fase 4: trayectoria lineal cartesiana ────────────────────────
    p_start = arm_model.forward_kinematics(arm._q)
    p_end   = p_start + np.array([0.04, 0.02, -0.015])
    qs_traj = arm_model.cartesian_trajectory(p_start, p_end, n_steps=45)
    for q_step in qs_traj:
        arm._q = q_step
        snap("Brazo: trayectoria lineal cartesiana")
    for _ in range(20):
        snap("Brazo: trayectoria lineal cartesiana")

    # ── Fase 5: giro de la base ──────────────────────────────────────
    # w_real = (r/L)·(0.8-(-0.8)) = (0.05/0.19)·1.6 ≈ 0.421 rad/s
    # 90° = π/2 rad  →  t = 3.74 s  →  ~75 pasos
    bot.set_twist(0.0, np.pi / 4)
    n_turn = int(np.ceil((np.pi / 2) / (0.421 * DT))) + 3
    for _ in range(n_turn):
        bot.step(DT)
        snap("Base: giro ~90 °  →  ω_real ≈ 0.421 rad/s")
    bot.omega[:] = 0.0

    # ── Fase 6: brazo → home ─────────────────────────────────────────
    q_from = arm._q.copy()
    q_to   = arm.q_home.copy()
    for t in np.linspace(0.0, 1.0, 60):
        arm._q = (1 - t) * q_from + t * q_to
        snap("Brazo → home  (posición final)")
    arm._q = q_to.copy()

    # ── Final estático ───────────────────────────────────────────────
    for _ in range(40):
        snap("Simulación completada")

    print(f"[SIM] Fotogramas generados: {len(frames)}")
    return frames, arm_model


# ══════════════════════════════════════════════════════════════════════
# Figura y animación
# ══════════════════════════════════════════════════════════════════════

def main():
    print("[SIM] Pre-computando fotogramas…")
    frames, arm_model = build_frames()
    total = len(frames)

    # ── Figura principal ─────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 7.2), facecolor=BG)
    fig.suptitle(
        "PuzzleBot  —  Simulación Visual",
        color=WHITE, fontsize=14, fontweight='bold', y=0.975,
    )

    # GridSpec: columna izquierda más ancha (3D), derecha más estrecha (2D)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.55, 1],
                  left=0.04, right=0.97, bottom=0.10, top=0.92, wspace=0.08)

    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax2d = fig.add_subplot(gs[1])

    # ── Estética 3D ──────────────────────────────────────────────────
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor(PANEL)
        pane.set_edgecolor(GRID)
    ax3d.set_facecolor(PANEL)
    ax3d.grid(True, color=GRID, linewidth=0.5)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.label.set_color(WHITE)
        axis.set_tick_params(labelcolor='#7799bb', labelsize=7)
    ax3d.set_xlabel('X (m)', labelpad=4)
    ax3d.set_ylabel('Y (m)', labelpad=4)
    ax3d.set_zlabel('Z (m)', labelpad=4)
    ax3d.set_xlim(-0.05, 0.50)
    ax3d.set_ylim(-0.22, 0.22)
    ax3d.set_zlim(0.00, 0.28)
    ax3d.set_title('Vista 3-D — brazo', color='#99aacc', fontsize=10, pad=6)
    ax3d.view_init(elev=22, azim=-48)

    # Plano del suelo (semi-transparente)
    _Xg = np.array([[-0.05, 0.50], [-0.05, 0.50]])
    _Yg = np.array([[-0.22, -0.22], [0.22, 0.22]])
    _Zg = np.zeros_like(_Xg)
    ax3d.plot_surface(_Xg, _Yg, _Zg, color=BLUE, alpha=0.06, zorder=0)

    # ── Artistas 3D ──────────────────────────────────────────────────
    link_lines = [
        ax3d.plot([], [], [], '-o',
                  color=c, linewidth=4, markersize=8,
                  markeredgewidth=0, solid_capstyle='round')[0]
        for c in LINK_CLR
    ]
    # Esfera en el EE
    ee_sphere, = ax3d.plot([], [], [], 'o',
                            color=WHITE, markersize=10,
                            markeredgecolor=PURPLE, markeredgewidth=1.5,
                            zorder=6)
    # Trail del EE
    ee_trail, = ax3d.plot([], [], [], '--',
                           color=PURPLE, linewidth=1.2, alpha=0.60)
    # Indicador de la base del robot
    base_mark, = ax3d.plot([], [], [], 's',
                            color=BLUE, markersize=11,
                            markeredgecolor=WHITE, markeredgewidth=1.0,
                            zorder=5)
    # Línea vertical eje del brazo
    base_axis, = ax3d.plot([], [], [], color='#334466',
                            linewidth=1.0, linestyle=':')

    # ── Estética 2D ──────────────────────────────────────────────────
    ax2d.set_facecolor(PANEL)
    ax2d.tick_params(colors='#7799bb', labelsize=8)
    for sp in ax2d.spines.values():
        sp.set_edgecolor(GRID)
    ax2d.set_xlabel('X (m)', color=WHITE, fontsize=9)
    ax2d.set_ylabel('Y (m)', color=WHITE, fontsize=9)
    ax2d.set_title('Vista cenital — base', color='#99aacc', fontsize=10)
    ax2d.set_xlim(-0.22, 0.52)
    ax2d.set_ylim(-0.32, 0.32)
    ax2d.set_aspect('equal')
    ax2d.grid(True, color=GRID, linewidth=0.5)
    ax2d.axhline(0, color=GRID, linewidth=0.9)
    ax2d.axvline(0, color=GRID, linewidth=0.9)

    # ── Artistas 2D ──────────────────────────────────────────────────
    body_patch = Polygon(robot_corners([0, 0, 0]), closed=True,
                         fc=BLUE, ec=WHITE, lw=1.6, alpha=0.72, zorder=3)
    ax2d.add_patch(body_patch)

    # Trayectoria recorrida por la base
    path_line, = ax2d.plot([], [], color=YELLOW, linewidth=1.8,
                            alpha=0.80, zorder=2)
    # Proyección XY del EE
    ee_proj2d, = ax2d.plot([], [], 'o', color=RED, markersize=9,
                            zorder=5, label='EE (proj. XY)', markeredgecolor=WHITE,
                            markeredgewidth=0.8)
    # Trail 2D del EE
    ee_trail2d, = ax2d.plot([], [], '--', color=PURPLE, linewidth=1.0,
                             alpha=0.55, zorder=2)
    # Flecha de orientación (heading)
    heading_line, = ax2d.plot([], [], color=WHITE, linewidth=2.5,
                               solid_capstyle='round', zorder=4)
    heading_tip,  = ax2d.plot([], [], '^', color=ORANGE, markersize=10,
                               zorder=6)
    # Ruedas (2 líneas perpendiculares a heading)
    wheel_l, = ax2d.plot([], [], '-', color='#556688', linewidth=4,
                          solid_capstyle='round', zorder=3)
    wheel_r, = ax2d.plot([], [], '-', color='#556688', linewidth=4,
                          solid_capstyle='round', zorder=3)

    ax2d.legend(loc='lower right', facecolor=PANEL, labelcolor=WHITE,
                edgecolor=GRID, fontsize=8, framealpha=0.9)

    # ── Barra de progreso ────────────────────────────────────────────
    progress_bg  = fig.add_axes([0.04, 0.04, 0.92, 0.018])
    progress_bar = fig.add_axes([0.04, 0.04, 0.00, 0.018])
    for ax in (progress_bg, progress_bar):
        ax.set_xticks([]); ax.set_yticks([])
    progress_bg.set_facecolor(GRID)
    progress_bar.set_facecolor(YELLOW)

    # ── Textos de estado ─────────────────────────────────────────────
    phase_txt = ax3d.text2D(0.03, 0.96, '', transform=ax3d.transAxes,
                             color=YELLOW, fontsize=9.5, va='top',
                             fontweight='bold')
    frame_txt = ax3d.text2D(0.97, 0.96, '', transform=ax3d.transAxes,
                             color='#556688', fontsize=7.5, va='top',
                             ha='right')
    info_txt = fig.text(
        0.50, 0.002, '',
        ha='center', va='bottom',
        color='#7799bb', fontsize=8.5, fontfamily='monospace',
        transform=fig.transFigure,
    )

    # ── Historial de trayectorias ─────────────────────────────────────
    TRAIL = 180
    ee_hist   = []
    pose_hist = []

    # ── Función de actualización ──────────────────────────────────────
    def update(fi):
        if fi == 0:
            ee_hist.clear()
            pose_hist.clear()

        f     = frames[fi]
        pose  = f['pose']
        q     = f['q']
        label = f['label']

        joints = arm_joints_world(pose, arm_model, q)
        ee     = joints[-1]
        ee_hist.append(ee.copy())
        pose_hist.append(pose.copy())

        # ────────────────────────── 3-D ──────────────────────────────
        for i, line in enumerate(link_lines):
            a, b = joints[i], joints[i + 1]
            line.set_data([a[0], b[0]], [a[1], b[1]])
            line.set_3d_properties([a[2], b[2]])

        ee_sphere.set_data([ee[0]], [ee[1]])
        ee_sphere.set_3d_properties([ee[2]])

        if len(ee_hist) > 1:
            tr = np.array(ee_hist[-TRAIL:])
            ee_trail.set_data(tr[:, 0], tr[:, 1])
            ee_trail.set_3d_properties(tr[:, 2])
            ee_trail2d.set_data(tr[:, 0], tr[:, 1])

        base_mark.set_data([pose[0]], [pose[1]])
        base_mark.set_3d_properties([0.0])

        # Eje vertical del brazo
        base_axis.set_data([pose[0], pose[0]], [pose[1], pose[1]])
        base_axis.set_3d_properties([0.0, arm_model.l1])

        # ────────────────────────── 2-D ──────────────────────────────
        body_patch.set_xy(robot_corners(pose))

        if len(pose_hist) > 1:
            ph = np.array(pose_hist)
            path_line.set_data(ph[:, 0], ph[:, 1])

        ee_proj2d.set_data([ee[0]], [ee[1]])

        # Heading arrow
        HL = 0.092
        hx = pose[0] + HL * np.cos(pose[2])
        hy = pose[1] + HL * np.sin(pose[2])
        heading_line.set_data([pose[0], hx], [pose[1], hy])
        heading_tip.set_data([hx], [hy])

        # Ruedas (líneas laterales perpendiculares al heading)
        WL, WD = 0.045, 0.072
        perp = pose[2] + np.pi / 2
        for wheel, sign in ((wheel_l, +1), (wheel_r, -1)):
            cx = pose[0] + sign * WD * np.sin(pose[2])
            cy = pose[1] - sign * WD * np.cos(pose[2])
            wx0 = cx + WL * np.cos(perp + np.pi / 2)
            wy0 = cy + WL * np.sin(perp + np.pi / 2)
            wx1 = cx - WL * np.cos(perp + np.pi / 2)
            wy1 = cy - WL * np.sin(perp + np.pi / 2)
            wheel.set_data([wx0, wx1], [wy0, wy1])

        # ── Textos ───────────────────────────────────────────────────
        phase_txt.set_text(label)
        frame_txt.set_text(f"frame {fi+1}/{total}")

        q_deg = np.degrees(q)
        info_txt.set_text(
            f"q = [{q_deg[0]:+6.1f}°  {q_deg[1]:+6.1f}°  {q_deg[2]:+6.1f}°]"
            f"    EE_mundo = ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m"
            f"    θ_base = {np.degrees(pose[2]):+.1f}°"
        )

        # Barra de progreso
        progress_bar.set_position([0.04, 0.04, 0.92 * (fi + 1) / total, 0.018])

        return (*link_lines, ee_sphere, ee_trail, base_mark, base_axis,
                body_patch, path_line, ee_proj2d, ee_trail2d,
                heading_line, heading_tip, wheel_l, wheel_r,
                phase_txt, frame_txt, info_txt)

    ani = animation.FuncAnimation(
        fig, update,
        frames=total,
        interval=50,        # 20 fps
        blit=False,
        repeat=True,
        repeat_delay=1500,
    )

    plt.show()
    return ani   # evita que el GC borre la animación antes de que cierre


if __name__ == '__main__':
    ani = main()
