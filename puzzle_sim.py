import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D         

from classes.puzzlebot.puzzlebot import PuzzleBot, PuzzleBotModel
from classes.puzzlebot.puzzlebot_arm import PuzzleBotArm, PuzzleBotArmModel

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

LINK_CLR = [RED, YELLOW, GREEN] 

def arm_joints_world(pose, arm_model, q):
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c23    = np.cos(q2 + q3)
    s23    = np.sin(q2 + q3)
    l1, l2, l3 = arm_model.l1, arm_model.l2, arm_model.l3

    pts_local = np.array([
        [0.0,               0.0,               0.0              ],
        [0.0,               0.0,               l1               ],
        [l2 * c2 * c1,      l2 * c2 * s1,      l1 + l2 * s2    ],
        [(l2*c2 + l3*c23)*c1, (l2*c2 + l3*c23)*s1, l1 + l2*s2 + l3*s23],
    ])

    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0., 0., 1.0]])
    t = np.array([x, y, 0.0])

    return (R @ pts_local.T).T + t


def robot_corners(pose, hw=0.085, hh=0.060):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s, c]])
    local = np.array([[ hh,  hw],
                      [ hh, -hw],
                      [-hh, -hw],
                      [-hh,  hw]])
    return (R2 @ local.T).T + np.array([x, y])

def build_frames():
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

    for _ in range(30):
        snap("Inicio")


    bot.set_twist(0.20, 0.0)
    for _ in range(100):
        bot.step(DT)
        snap("Base: avance recto  →  v_real ≈ 0.04 m/s")

    q_from = arm._q.copy()
    q_to   = arm.q_home.copy()
    for t in np.linspace(0.0, 1.0, 50):
        arm._q = (1 - t) * q_from + t * q_to
        snap("Brazo → home  [q1=0°  q2=30°  q3=-45°]")
    arm._q = q_to.copy()

    p_goal = np.array([0.08, 0.04, 0.16])
    q_goal = arm_model.inverse_kinematics(p_goal)
    q_from = arm._q.copy()
    for t in np.linspace(0.0, 1.0, 60):
        arm._q = (1 - t) * q_from + t * q_goal
        snap(f"Brazo → IK  target=({p_goal[0]:.2f},{p_goal[1]:.2f},{p_goal[2]:.2f}) m")
    arm._q = q_goal.copy()

    for _ in range(15):
        snap(f"Brazo → IK  target=({p_goal[0]:.2f},{p_goal[1]:.2f},{p_goal[2]:.2f}) m")

    p_start = arm_model.forward_kinematics(arm._q)
    p_end   = p_start + np.array([0.04, 0.02, -0.015])
    qs_traj = arm_model.cartesian_trajectory(p_start, p_end, n_steps=45)
    for q_step in qs_traj:
        arm._q = q_step
        snap("Brazo: trayectoria lineal cartesiana")
    for _ in range(20):
        snap("Brazo: trayectoria lineal cartesiana")

    bot.set_twist(0.0, np.pi / 4)
    n_turn = int(np.ceil((np.pi / 2) / (0.421 * DT))) + 3
    for _ in range(n_turn):
        bot.step(DT)
        snap("Base: giro ~90 °  →  ω_real ≈ 0.421 rad/s")
    bot.omega[:] = 0.0

    q_from = arm._q.copy()
    q_to   = arm.q_home.copy()
    for t in np.linspace(0.0, 1.0, 60):
        arm._q = (1 - t) * q_from + t * q_to
        snap("Brazo → home  (posición final)")
    arm._q = q_to.copy()

    for _ in range(40):
        snap("Simulación completada")

    print(f"[SIM] Fotogramas generados: {len(frames)}")
    return frames, arm_model

def main():
    print("[SIM] Pre-computando fotogramas…")
    frames, arm_model = build_frames()
    total = len(frames)

    fig = plt.figure(figsize=(15, 7.2), facecolor=BG)
    fig.suptitle(
        "PuzzleBot  —  Simulación Visual",
        color=WHITE, fontsize=14, fontweight='bold', y=0.975,
    )

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.55, 1],
                  left=0.04, right=0.97, bottom=0.10, top=0.92, wspace=0.08)

    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax2d = fig.add_subplot(gs[1])

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

    _Xg = np.array([[-0.05, 0.50], [-0.05, 0.50]])
    _Yg = np.array([[-0.22, -0.22], [0.22, 0.22]])
    _Zg = np.zeros_like(_Xg)
    ax3d.plot_surface(_Xg, _Yg, _Zg, color=BLUE, alpha=0.06, zorder=0)

    link_lines = [
        ax3d.plot([], [], [], '-o',
                  color=c, linewidth=4, markersize=8,
                  markeredgewidth=0, solid_capstyle='round')[0]
        for c in LINK_CLR
    ]
    ee_sphere, = ax3d.plot([], [], [], 'o',
                            color=WHITE, markersize=10,
                            markeredgecolor=PURPLE, markeredgewidth=1.5,
                            zorder=6)
    ee_trail, = ax3d.plot([], [], [], '--',
                           color=PURPLE, linewidth=1.2, alpha=0.60)
    base_mark, = ax3d.plot([], [], [], 's',
                            color=BLUE, markersize=11,
                            markeredgecolor=WHITE, markeredgewidth=1.0,
                            zorder=5)
    base_axis, = ax3d.plot([], [], [], color='#334466',
                            linewidth=1.0, linestyle=':')

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

    body_patch = Polygon(robot_corners([0, 0, 0]), closed=True,
                         fc=BLUE, ec=WHITE, lw=1.6, alpha=0.72, zorder=3)
    ax2d.add_patch(body_patch)

    path_line, = ax2d.plot([], [], color=YELLOW, linewidth=1.8,
                            alpha=0.80, zorder=2)
    ee_proj2d, = ax2d.plot([], [], 'o', color=RED, markersize=9,
                            zorder=5, label='EE (proj. XY)', markeredgecolor=WHITE,
                            markeredgewidth=0.8)
    ee_trail2d, = ax2d.plot([], [], '--', color=PURPLE, linewidth=1.0,
                             alpha=0.55, zorder=2)
    heading_line, = ax2d.plot([], [], color=WHITE, linewidth=2.5,
                               solid_capstyle='round', zorder=4)
    heading_tip,  = ax2d.plot([], [], '^', color=ORANGE, markersize=10,
                               zorder=6)
    wheel_l, = ax2d.plot([], [], '-', color='#556688', linewidth=4,
                          solid_capstyle='round', zorder=3)
    wheel_r, = ax2d.plot([], [], '-', color='#556688', linewidth=4,
                          solid_capstyle='round', zorder=3)

    ax2d.legend(loc='lower right', facecolor=PANEL, labelcolor=WHITE,
                edgecolor=GRID, fontsize=8, framealpha=0.9)

    progress_bg  = fig.add_axes([0.04, 0.04, 0.92, 0.018])
    progress_bar = fig.add_axes([0.04, 0.04, 0.00, 0.018])
    for ax in (progress_bg, progress_bar):
        ax.set_xticks([]); ax.set_yticks([])
    progress_bg.set_facecolor(GRID)
    progress_bar.set_facecolor(YELLOW)

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

    _DYN_BG   = '#0d0d1a'
    _DYN_GRID = '#1c2b42'
    fig2, axes2 = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    fig2.patch.set_facecolor(_DYN_BG)
    fig2.suptitle('Dinámica del sistema — PuzzleBot',
                  color=WHITE, fontsize=13, fontweight='bold')

    _dyn_titles = [
        'Ángulos de juntas  q1, q2, q3  (°)',
        'Posición EE  x, y, z  (m)',
        'Posición base  x, y  (m)',
        'Orientación base  θ  (°)',
    ]
    for i, title in enumerate(_dyn_titles):
        a = axes2[i]
        a.set_facecolor(_DYN_BG)
        a.set_title(title, color='#aaaaaa', fontsize=9, loc='left', pad=3)
        a.tick_params(colors='#555')
        for sp in a.spines.values():
            sp.set_color('#333')
        a.grid(True, color=_DYN_GRID, linewidth=0.6)
    axes2[-1].set_xlabel('Tiempo  (frame × 0.05 s)', color='#aaaaaa')

    dyn_q1, = axes2[0].plot([], [], color=RED,    lw=1.4, label='q1')
    dyn_q2, = axes2[0].plot([], [], color=YELLOW, lw=1.4, label='q2')
    dyn_q3, = axes2[0].plot([], [], color=GREEN,  lw=1.4, label='q3')
    axes2[0].legend(facecolor='#161b22', edgecolor='#333',
                    labelcolor='white', fontsize=8, loc='upper right')

    dyn_ex, = axes2[1].plot([], [], color=BLUE,   lw=1.4, label='x')
    dyn_ey, = axes2[1].plot([], [], color=GREEN,  lw=1.4, label='y')
    dyn_ez, = axes2[1].plot([], [], color=PURPLE, lw=1.4, label='z')
    axes2[1].legend(facecolor='#161b22', edgecolor='#333',
                    labelcolor='white', fontsize=8, loc='upper right')

    dyn_bx, = axes2[2].plot([], [], color=BLUE,   lw=1.4, label='x')
    dyn_by, = axes2[2].plot([], [], color=ORANGE, lw=1.4, label='y')
    axes2[2].legend(facecolor='#161b22', edgecolor='#333',
                    labelcolor='white', fontsize=8, loc='upper right')

    dyn_th, = axes2[3].plot([], [], color=WHITE, lw=1.4)

    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    t_dyn                     = []
    q1_dyn, q2_dyn, q3_dyn   = [], [], []
    ex_dyn, ey_dyn, ez_dyn   = [], [], []
    bx_dyn, by_dyn, th_dyn   = [], [], []

    TRAIL = 180
    ee_hist   = []
    pose_hist = []

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

        base_axis.set_data([pose[0], pose[0]], [pose[1], pose[1]])
        base_axis.set_3d_properties([0.0, arm_model.l1])

        body_patch.set_xy(robot_corners(pose))

        if len(pose_hist) > 1:
            ph = np.array(pose_hist)
            path_line.set_data(ph[:, 0], ph[:, 1])

        ee_proj2d.set_data([ee[0]], [ee[1]])

        HL = 0.092
        hx = pose[0] + HL * np.cos(pose[2])
        hy = pose[1] + HL * np.sin(pose[2])
        heading_line.set_data([pose[0], hx], [pose[1], hy])
        heading_tip.set_data([hx], [hy])

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

        phase_txt.set_text(label)
        frame_txt.set_text(f"frame {fi+1}/{total}")

        q_deg = np.degrees(q)
        info_txt.set_text(
            f"q = [{q_deg[0]:+6.1f}°  {q_deg[1]:+6.1f}°  {q_deg[2]:+6.1f}°]"
            f"    EE_mundo = ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) m"
            f"    θ_base = {np.degrees(pose[2]):+.1f}°"
        )

        progress_bar.set_position([0.04, 0.04, 0.92 * (fi + 1) / total, 0.018])

        # ── Dinámica del sistema ──────────────────────────────────────
        if fi == 0:
            t_dyn.clear()
            for lst in (q1_dyn, q2_dyn, q3_dyn,
                        ex_dyn, ey_dyn, ez_dyn,
                        bx_dyn, by_dyn, th_dyn):
                lst.clear()

        q_deg = np.degrees(q)
        t_dyn.append(fi * 0.05)
        q1_dyn.append(q_deg[0]);  q2_dyn.append(q_deg[1]);  q3_dyn.append(q_deg[2])
        ex_dyn.append(ee[0]);     ey_dyn.append(ee[1]);     ez_dyn.append(ee[2])
        bx_dyn.append(pose[0]);   by_dyn.append(pose[1])
        th_dyn.append(np.degrees(pose[2]))

        dyn_q1.set_data(t_dyn, q1_dyn)
        dyn_q2.set_data(t_dyn, q2_dyn)
        dyn_q3.set_data(t_dyn, q3_dyn)
        dyn_ex.set_data(t_dyn, ex_dyn)
        dyn_ey.set_data(t_dyn, ey_dyn)
        dyn_ez.set_data(t_dyn, ez_dyn)
        dyn_bx.set_data(t_dyn, bx_dyn)
        dyn_by.set_data(t_dyn, by_dyn)
        dyn_th.set_data(t_dyn, th_dyn)

        for a in axes2:
            a.relim()
            a.autoscale_view()
        fig2.canvas.draw_idle()

        return (*link_lines, ee_sphere, ee_trail, base_mark, base_axis,
                body_patch, path_line, ee_proj2d, ee_trail2d,
                heading_line, heading_tip, wheel_l, wheel_r,
                phase_txt, frame_txt, info_txt)

    ani = animation.FuncAnimation(
        fig, update,
        frames=total,
        interval=50,        
        blit=False,
        repeat=True,
        repeat_delay=1500,
    )

    plt.show()
    return ani


if __name__ == '__main__':
    ani = main()
