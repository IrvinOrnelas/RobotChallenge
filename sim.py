"""
sim.py
2D Matplotlib simulator for the RobotChallenge warehouse scenario.
Renders all robots, boxes, zones and trails in real time.

Run:
    python sim.py
    python sim.py --speed 5      # 5× real-time
    python sim.py --no-lidar     # hide LiDAR rays
    python sim.py --save         # save MP4 instead of live display
"""
import sys, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, '.')
from utils import norm2, rad2deg
from classes.husky.husky import Husky
from classes.husky.lidar import LiDAR
from classes.anymal.anymal import Anymal
from classes.puzzlebot.puzzlebot import PuzzleBot, PuzzleBotModel
from classes.puzzlebot.puzzlebot_arm import PuzzleBotArm, PuzzleBotArmModel
from coordinator import Coordinator, Box

# ── WORLD CONSTANTS ──────────────────────────────────────────────────────────
WORLD_W, WORLD_H = 14.0, 6.0
DT = 0.05           # simulation timestep (s)
DRAW_EVERY = 1      # draw every N simulation steps

# ── COLORS ───────────────────────────────────────────────────────────────────
C_BG       = '#0a0f1e'
C_GRID     = '#1e293b'
C_CORRIDOR = '#ef444422'
C_STACK    = '#10b98122'
C_HUSKY    = '#0ea5e9'
C_ANYMAL   = '#7c3aed'
C_PB       = ['#fbbf24', '#34d399', '#f472b6']
C_BOX      = {'A': '#ef4444', 'B': '#f59e0b', 'C': '#10b981'}
C_LIDAR    = '#06b6d422'
C_TEXT     = '#e2e8f0'
C_DIM      = '#475569'


def build_robots():
    """Instantiate all robots and return coordinator."""
    husky = Husky(pose=(2.0, -1.8, 0.0))

    anymal = Anymal(pose=(0.0, 0.0, 0.0), payload_kg=6.0)

    arm_model = PuzzleBotArmModel(l1=0.10, l2=0.08, l3=0.06)
    puzzlebots = [
        PuzzleBot(i, pose=(9.5 + i * 0.5, 2.5, 0.0),
                  arm=PuzzleBotArm(PuzzleBotArmModel()))
        for i in range(3)
    ]
    
    obstacle_boxes = [
        Box('B1', 2.5,  0.5, w=0.5, h=0.5, color='#8B4513', obstacle_box=True),
        Box('B2', 3.5, -0.4, w=0.5, h=0.5, color='#8B4513', obstacle_box=True),
        Box('B3', 4.5,  0.3, w=0.5, h=0.5, color='#8B4513', obstacle_box=True),
    ]
    # Small boxes in work zone — PuzzleBots stack these
    stack_boxes = [
        Box('A', 10.0,  1.5, w=0.35, h=0.35, color='#ef4444'),
        Box('B',  9.5,  3.0, w=0.35, h=0.35, color='#f59e0b'),
        Box('C', 11.0,  2.0, w=0.35, h=0.35, color='#10b981'),
    ]

    coord = Coordinator(husky, anymal, puzzlebots,
                        obstacle_boxes, stack_boxes,
                        anymal_dest=(10.5, 0),
                        stack_target=(12.0, 0.5))
    return coord


class Sim2D:
    """
    2D top-down simulation with matplotlib FuncAnimation.
    """

    def __init__(self, speed: float = 1.0, show_lidar: bool = True,
                 show_trails: bool = True, save: bool = False):
        self.speed = speed
        self.show_lidar = show_lidar
        self.show_trails = show_trails
        self.save = save
        self.coord = build_robots()
        self.lidar = LiDAR(n_rays=72, max_range=5.0, noise_std=0.01)
        self.t = 0.0
        self.step_count = 0

        # ── figure setup ────────────────────────────────────────────────────
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(
            1, 2,
            figsize=(16, 7),
            gridspec_kw={'width_ratios': [3, 1]},
            facecolor=C_BG
        )
        self.ax   = self.axes[0]   # main simulation view
        self.ax_i = self.axes[1]   # info panel

        self._setup_world()
        self._setup_artists()
        self._setup_info_panel()

        self.fig.tight_layout(pad=1.5)

    # ── WORLD SETUP ──────────────────────────────────────────────────────────

    def _setup_world(self):
        ax = self.ax
        ax.set_facecolor(C_BG)
        ax.set_xlim(-0.5, WORLD_W + 0.5)
        ax.set_ylim(-WORLD_H/2 - 0.5, WORLD_H/2 + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', color=C_DIM, fontsize=8)
        ax.set_ylabel('y (m)', color=C_DIM, fontsize=8)
        ax.tick_params(colors=C_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)

        # Grid
        for x in range(int(WORLD_W) + 1):
            ax.axvline(x, color=C_GRID, lw=0.4, zorder=0)
        for y in np.arange(-WORLD_H/2, WORLD_H/2 + 1, 1):
            ax.axhline(y, color=C_GRID, lw=0.4, zorder=0)

        # Corridor zone (to be cleared)
        corr = patches.Rectangle((0, -1), 6, 2,
                                  fc=C_CORRIDOR, ec='#ef4444', lw=1,
                                  ls='--', zorder=1)
        ax.add_patch(corr)
        ax.text(0.2, -0.9, 'CLEAR ZONE', color='#ef4444', fontsize=7,
                fontfamily='monospace', zorder=2)

        # Stack target zone
        stack = patches.Rectangle((11.5, 0), 2, 2,
                                   fc=C_STACK, ec='#10b981', lw=1,
                                   ls='--', zorder=1)
        ax.add_patch(stack)
        ax.text(11.55, 2.1, 'STACK', color='#10b981', fontsize=7,
                fontfamily='monospace', zorder=2)

        # ANYmal destination
        ax.plot(11.0, 3.6, 'o', ms=8, mfc='none', mec='#a78bfa', mew=1.5, zorder=3)
        ax.text(11.1, 3.65, 'p_dest', color='#a78bfa', fontsize=7,
                fontfamily='monospace', zorder=3)

    # ── ARTIST SETUP ────────────────────────────────────────────────────────

    def _setup_artists(self):
        ax = self.ax
        c = self.coord

        # ── Trails ──────────────────────────────────────────────────────────
        self.husky_trail,  = ax.plot([], [], '-', color=C_HUSKY,   lw=0.8, alpha=0.4, zorder=2)
        self.anymal_trail, = ax.plot([], [], '-', color=C_ANYMAL,  lw=0.8, alpha=0.4, zorder=2)
        self.pb_trails = [
            ax.plot([], [], '-', color=C_PB[i], lw=0.8, alpha=0.4, zorder=2)[0]
            for i in range(3)
        ]

        # ── LiDAR rays ──────────────────────────────────────────────────────
        self.lidar_lines = [
            ax.plot([], [], '-', color=C_LIDAR, lw=0.3, zorder=2)[0]
            for _ in range(self.lidar.n_rays)
        ]

        # ── Boxes ────────────────────────────────────────────────────────────
        self.box_patches = {}
        self.box_texts = {}
        for box in c.boxes:
            p = patches.Rectangle(
                (box.x - box.w/2, box.y - box.h/2), box.w, box.h,
                fc=C_BOX.get(box.id, '#94a3b8'), ec='white', lw=1.2, zorder=4
            )
            ax.add_patch(p)
            t = ax.text(box.x, box.y, box.id, color='white',
                        ha='center', va='center', fontsize=8,
                        fontfamily='monospace', fontweight='bold', zorder=5)
            self.box_patches[box.id] = p
            self.box_texts[box.id] = t

        # ── Husky (rectangle) ────────────────────────────────────────────────
        self.husky_body = patches.FancyBboxPatch(
            (-0.28, -0.18), 0.56, 0.36,
            boxstyle='round,pad=0.02',
            fc=C_HUSKY, ec='#7dd3fc', lw=1.5, zorder=5
        )
        ax.add_patch(self.husky_body)
        self.husky_arrow = ax.annotate(
            '', xy=(0.35, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5), zorder=6
        )
        self.husky_label = ax.text(0, 0, 'HUSKY', color='#7dd3fc',
                                   fontsize=6, fontfamily='monospace', zorder=6)

        # ── ANYmal (ellipse + legs) ────────────────────────────────────────
        self.anymal_body = patches.Ellipse(
            (0, 0), 0.7, 0.4,
            fc=C_ANYMAL, ec='#c4b5fd', lw=1.5, zorder=5
        )
        ax.add_patch(self.anymal_body)
        self.anymal_legs = [ax.plot([], [], '-o', color='#a78bfa', lw=1.5,
                                    ms=3, zorder=5)[0] for _ in range(4)]
        self.anymal_label = ax.text(0, 0, 'ANYmal', color='#c4b5fd',
                                    fontsize=6, fontfamily='monospace', zorder=6)

        # ── PuzzleBots ────────────────────────────────────────────────────
        self.pb_bodies = [
            patches.Circle((0, 0), 0.15, fc=C_PB[i], ec='white', lw=1, zorder=5)
            for i in range(3)
        ]
        self.pb_arms = [
            ax.plot([], [], '-', color='white', lw=1.5, zorder=6)[0]
            for _ in range(3)
        ]
        self.pb_labels = [
            ax.text(0, 0, f'PB{i}', color=C_PB[i], fontsize=6,
                    fontfamily='monospace', zorder=6)
            for i in range(3)
        ]
        for p in self.pb_bodies:
            ax.add_patch(p)

        # ── Time / Phase text ──────────────────────────────────────────────
        self.time_text  = ax.text(0.01, 0.99, '', transform=ax.transAxes,
                                   color=C_TEXT, fontsize=8, fontfamily='monospace',
                                   va='top', zorder=10)
        self.phase_text = ax.text(0.01, 0.93, '', transform=ax.transAxes,
                                   color='#f59e0b', fontsize=8, fontfamily='monospace',
                                   va='top', zorder=10)

    # ── INFO PANEL ───────────────────────────────────────────────────────────

    def _setup_info_panel(self):
        ax = self.ax_i
        ax.set_facecolor(C_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')

        ax.text(0.05, 0.97, 'TELEMETRY', color='#64748b', fontsize=8,
                fontfamily='monospace', va='top', fontweight='bold')
        ax.text(0.05, 0.92, '─' * 22, color='#1e293b', fontsize=7,
                fontfamily='monospace', va='top')

        self.info_texts = {}
        labels = [
            ('husky_pos',  'Husky pos',    0.88),
            ('husky_v',    'Husky v_cmd',  0.83),
            ('husky_vmeas','Husky v_meas', 0.78),
            ('husky_slip', 'Husky slip s', 0.73),
            ('sep1',       '─' * 22,       0.68),
            ('any_pos',    'ANYmal pos',   0.63),
            ('any_dist',   'dist→goal',    0.58),
            ('any_phase',  'gait phase',   0.53),
            ('sep2',       '─' * 22,       0.48),
            ('pb0_state',  'PB0 state',    0.43),
            ('pb1_state',  'PB1 state',    0.38),
            ('pb2_state',  'PB2 state',    0.33),
            ('sep3',       '─' * 22,       0.28),
            ('detJ',       '|det J| PB0',  0.23),
        ]
        for key, lbl, y in labels:
            if key.startswith('sep'):
                t = ax.text(0.05, y, lbl, color='#1e293b',
                            fontsize=6, fontfamily='monospace', va='top')
            else:
                ax.text(0.05, y, lbl + ':', color='#64748b',
                        fontsize=7, fontfamily='monospace', va='top')
                t = ax.text(0.55, y, '—', color=C_TEXT,
                            fontsize=7, fontfamily='monospace', va='top')
            self.info_texts[key] = t

        # Legend
        legend_y = 0.14
        ax.text(0.05, legend_y, 'LEGEND', color='#64748b', fontsize=7,
                fontfamily='monospace', va='top', fontweight='bold')
        items = [('Husky', C_HUSKY), ('ANYmal', C_ANYMAL),
                 ('PB 0', C_PB[0]), ('PB 1', C_PB[1]), ('PB 2', C_PB[2])]
        for j, (lbl, col) in enumerate(items):
            ax.add_patch(patches.Rectangle(
                (0.05, legend_y - 0.04 - j*0.04), 0.06, 0.025,
                fc=col, ec='none', transform=ax.transAxes, zorder=5
            ))
            ax.text(0.14, legend_y - 0.025 - j*0.04, lbl, color=col,
                    fontsize=6, fontfamily='monospace', va='top')

    # ── UPDATE FRAME ─────────────────────────────────────────────────────────

    def _update_info(self):
        c = self.coord
        h = c.husky
        a = c.anymal
        pbs = c.puzzlebots
        pb_fsm = c.pb_fsm

        self.info_texts['husky_pos'].set_text(
            f'({h.x:.2f}, {h.y:.2f})')
        self.info_texts['husky_v'].set_text(f'{h.v_cmd:.3f} m/s')
        self.info_texts['husky_vmeas'].set_text(f'{h.v_meas:.3f} m/s')
        self.info_texts['husky_slip'].set_text(f'{h.model.s:.2f}')

        dist = norm2(c.anymal_dest[0]-a.x, c.anymal_dest[1]-a.y)
        self.info_texts['any_pos'].set_text(f'({a.x:.2f}, {a.y:.2f})')
        self.info_texts['any_dist'].set_text(f'{dist:.3f} m')
        self.info_texts['any_phase'].set_text(f'{a.gait_phase:.2f} rad')

        for i in range(3):
            state = pb_fsm[i].name if i < len(pb_fsm) else '—'
            self.info_texts[f'pb{i}_state'].set_text(state)

        # Jacobian determinant of PB0
        pb0 = pbs[0]
        detJ = pb0.arm.model.det_jacobian(*pb0.arm.q)
        color = '#ef4444' if abs(detJ) < 1e-3 else '#10b981'
        self.info_texts['detJ'].set_text(f'{detJ:.5f}')
        self.info_texts['detJ'].set_color(color)

    def _update_frame(self, frame):
        c = self.coord

        # ── Advance simulation ───────────────────────────────────────────────
        steps = max(1, int(self.speed))
        for _ in range(steps):
            if not c.task_completed():
                c.step(DT)

        # ── Update trails ───────────────────────────────────────────────────
        if self.show_trails and len(c.husky.trail) > 1:
            tr = np.array(c.husky.trail)
            self.husky_trail.set_data(tr[:,0], tr[:,1])
        if self.show_trails and len(c.anymal.trail) > 1:
            tr = np.array(c.anymal.trail)
            self.anymal_trail.set_data(tr[:,0], tr[:,1])
        for i, bot in enumerate(c.puzzlebots):
            if self.show_trails and len(bot.trail) > 1:
                tr = np.array(bot.trail)
                self.pb_trails[i].set_data(tr[:,0], tr[:,1])

        # ── LiDAR ────────────────────────────────────────────────────────────
        if self.show_lidar and c.phase.name == 'CLEARING':
            obstacles = [{'x': b.x, 'y': b.y, 'w': b.w, 'h': b.h}
                         for b in c.boxes]
            ranges, angles = self.lidar.scan(c.husky.pose, obstacles)
            pts = self.lidar.get_points(c.husky.pose, ranges, angles)
            for i, (line, pt) in enumerate(zip(self.lidar_lines, pts)):
                line.set_data([c.husky.x, pt[0]], [c.husky.y, pt[1]])
        else:
            for line in self.lidar_lines:
                line.set_data([], [])

        # ── Boxes ───────────────────────────────────────────────────────────
        for box in c.boxes:
            p = self.box_patches[box.id]
            p.set_xy((box.x - box.w/2, box.y - box.h/2))
            if box.stacked:
                p.set_alpha(0.6)
            self.box_texts[box.id].set_position((box.x, box.y))

        # ── Husky ───────────────────────────────────────────────────────────
        h = c.husky
        t_mat = matplotlib.transforms.Affine2D().rotate(h.theta).translate(h.x, h.y)
        self.husky_body.set_transform(t_mat + self.ax.transData)
        # arrow
        ax_end_x = h.x + 0.4 * np.cos(h.theta)
        ax_end_y = h.y + 0.4 * np.sin(h.theta)
        self.husky_arrow.set_position((h.x, h.y))
        self.husky_arrow.xy = (ax_end_x, ax_end_y)
        self.husky_arrow.xytext = (h.x, h.y)
        self.husky_label.set_position((h.x + 0.35, h.y + 0.25))

        # ── ANYmal ──────────────────────────────────────────────────────────
        a = c.anymal
        t_mat_a = matplotlib.transforms.Affine2D().rotate(a.theta).translate(a.x, a.y)
        self.anymal_body.set_transform(t_mat_a + self.ax.transData)
        leg_names = list(a.legs.keys())
        leg_offsets = [np.array([0.3, 0.2]), np.array([0.3,-0.2]),
                       np.array([-0.3, 0.2]), np.array([-0.3,-0.2])]
        for li, (leg_line, off) in enumerate(zip(self.anymal_legs, leg_offsets)):
            phase_i = a.gait_phase + li * np.pi/2
            hip_x = a.x + off[0]*np.cos(a.theta) - off[1]*np.sin(a.theta)
            hip_y = a.y + off[0]*np.sin(a.theta) + off[1]*np.cos(a.theta)
            foot_x = hip_x + 0.2 * np.sin(phase_i) * np.cos(a.theta)
            foot_y = hip_y + 0.2 * np.sin(phase_i) * np.sin(a.theta)
            leg_line.set_data([hip_x, foot_x], [hip_y, foot_y])
        self.anymal_label.set_position((a.x + 0.42, a.y + 0.25))

        # ── PuzzleBots ───────────────────────────────────────────────────────
        for i, bot in enumerate(c.puzzlebots):
            self.pb_bodies[i].center = (bot.x, bot.y)
            ee = bot.arm.ee_position()
            arm_ex = bot.x + ee[0] * np.cos(bot.theta) * 1.5
            arm_ey = bot.y + ee[0] * np.sin(bot.theta) * 1.5
            self.pb_arms[i].set_data([bot.x, arm_ex], [bot.y, arm_ey])
            self.pb_labels[i].set_position((bot.x + 0.18, bot.y + 0.18))

        # ── HUD ────────────────────────────────────────────────────────────
        self.time_text.set_text(f't = {c.t:.1f}s')
        self.phase_text.set_text(f'Phase: {c.phase.name}')

        # ── Info panel ─────────────────────────────────────────────────────
        self._update_info()

        return (self.husky_trail, self.anymal_trail,
                self.husky_body, self.anymal_body,
                self.time_text, self.phase_text,
                *self.pb_bodies, *self.pb_arms,
                *self.pb_trails, *self.lidar_lines,
                *list(self.box_patches.values()),
                *list(self.box_texts.values()),
                *list(self.info_texts.values()))

    # ── RUN ─────────────────────────────────────────────────────────────────

    def run(self):
        ani = animation.FuncAnimation(
            self.fig, self._update_frame,
            interval=50,          # ms between frames
            blit=True,
            cache_frame_data=False
        )

        if self.save:
            print("Saving animation to robot_sim.mp4 ...")
            writer = animation.FFMpegWriter(fps=20, bitrate=1800)
            ani.save('robot_sim.mp4', writer=writer)
            print("Saved.")
        else:
            plt.title('TE3002B — Almacén Robótico Colaborativo',
                      color=C_TEXT, fontfamily='monospace', fontsize=10, pad=8)
            plt.show()


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RobotChallenge 2D Simulator')
    parser.add_argument('--speed',    type=float, default=1.0,
                        help='Simulation speed multiplier (default 1.0)')
    parser.add_argument('--no-lidar', action='store_true',
                        help='Disable LiDAR ray visualization')
    parser.add_argument('--no-trails',action='store_true',
                        help='Disable robot trails')
    parser.add_argument('--save',     action='store_true',
                        help='Save animation as robot_sim.mp4')
    args = parser.parse_args()

    sim = Sim2D(
        speed=args.speed,
        show_lidar=not args.no_lidar,
        show_trails=not args.no_trails,
        save=args.save
    )
    sim.run()
