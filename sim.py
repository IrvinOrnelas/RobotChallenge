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
import sys, os, argparse
import numpy as np
import matplotlib

# If a display is available, prefer an interactive backend instead of Agg.
if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
    if matplotlib.get_backend().lower() == 'agg':
        for candidate in ['qtagg', 'gtk4agg', 'gtk3agg', 'tkagg', 'qt5agg', 'wxagg']:
            try:
                matplotlib.use(candidate, force=True)
                if matplotlib.get_backend().lower() != 'agg':
                    break
            except Exception:
                continue

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.draw import line as draw_line

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, '.')
from utils import norm2, rad2deg
from classes.husky.husky import Husky
from classes.husky.lidar import LiDAR
from classes.anymal.anymal import Anymal
from classes.puzzlebot.puzzlebot import PuzzleBot, PuzzleBotModel
from classes.puzzlebot.puzzlebot_arm import PuzzleBotArm, PuzzleBotArmModel
from classes.xram.lite_xarm import Lite6Arm
from coordinator import Coordinator, Box, Zone

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
        self.lidar = LiDAR(n_rays=144, max_range=5.0, noise_std=0.01)
        self.t = 0.0
        self.step_count = 0

        # ── figure setup ────────────────────────────────────────────────────
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(22, 7), facecolor=C_BG, constrained_layout=True)
        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            width_ratios=[3, 1.1, 1],
            height_ratios=[1, 1],
            hspace=0.08, wspace=0.22
        )
        self.ax     = self.fig.add_subplot(gs[:, 0])   # world (full height, left)
        self.ax_cam = self.fig.add_subplot(gs[0, 1])   # camera feed (top centre-right)
        self.ax_met = self.fig.add_subplot(gs[1, 1])   # metrics (bottom centre-right)
        self.ax_i   = self.fig.add_subplot(gs[:, 2])   # telemetry (full height, right)

        self._setup_world()
        self.coord = self.build_robots()
        self._setup_artists()
        self._setup_camera_panel()
        self._setup_metrics_panel()
        self._setup_info_panel()

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
            
        # Define our Zones
        self.clear_zone = Zone('CLEAR ZONE', x_min=1.0, x_max=7.0, y_min=-1.0, y_max=1.0, color='#ef4444')
        self.stack_zone = Zone('STACK ZONE', x_min=8.5, x_max=11.5, y_min=-2.0, y_max=2.0, color='#10b981')

        # ANYmal lane lines: center (white) + 2 laterals (yellow), x∈[0,8]
        ax.plot([0, 8], [0,    0   ], '--', color='#ffffff', lw=0.9,
                alpha=0.55, zorder=1, dashes=(6, 4))
        ax.plot([0, 8], [ 0.5,  0.5], '--', color='#fbbf24', lw=0.7,
                alpha=0.45, zorder=1, dashes=(4, 5))
        ax.plot([0, 8], [-0.5, -0.5], '--', color='#fbbf24', lw=0.7,
                alpha=0.45, zorder=1, dashes=(4, 5))

        # ANYmal destination
        ax.plot(8.0, 0, 'o', ms=8, mfc='none', mec='#a78bfa', mew=1.5, zorder=3)
        ax.text(8.1, 0.05, 'p_dest', color='#a78bfa', fontsize=7,
                fontfamily='monospace', zorder=3)
    
     # ── BUILD ROBOTS ────────────────────────────────────────────────────────
    def build_robots(self):
        """Instantiate all robots and return coordinator."""
        husky = Husky(pose=(1.3, -1.8, 0.0))

        anymal = Anymal(pose=(0.0, 0.0, 0.0), payload_kg=6.0)

        arm_model = PuzzleBotArmModel(l1=0.10, l2=0.08, l3=0.06)
        puzzlebots = [
            PuzzleBot(i, pose=(9.5, 2.5 + i * 0.5, 0.0),
                    arm=PuzzleBotArm(PuzzleBotArmModel()))
            for i in range(3)
        ]
        
        obstacle_boxes = [
            Box('B1', 2.0,  0.0, w=0.5, h=0.5, color='#8B4513', obstacle_box=True),
            Box('B2', 4.0,  0.0, w=0.5, h=0.5, color='#8B4513', obstacle_box=True),
            Box('B3', 6.0,  0.0, w=0.5, h=0.5, color='#8B4513', obstacle_box=True),
        ]
        # Small boxes in work zone — PuzzleBots stack these
        stack_boxes = [
            Box('A', 10.0,  -1.5, w=0.35, h=0.35, color='#ef4444'),
            Box('B', 10.0,  0, w=0.35, h=0.35, color='#f59e0b'),
            Box('C', 10.0,  1.5, w=0.35, h=0.35, color='#10b981'),
        ]
        
        lite6_arms = [
            Lite6Arm(0, 8.5,  0.8),
            Lite6Arm(1, 8.5, -0.8)
        ]

        coord = Coordinator(husky, anymal, puzzlebots,
                            obstacle_boxes, stack_boxes,
                            self.clear_zone, self.stack_zone,
                            xarms=lite6_arms,
                            anymal_dest=(8.0, 0),
                            stack_target=(10.0, 0.5),
                            lidar=self.lidar)
        return coord

    # ── ARTIST SETUP ────────────────────────────────────────────────────────

    def _setup_artists(self):
        ax = self.ax
        c = self.coord
        self.visual_artists = []
        
        # 1. Setup Zones
        self.visual_artists.extend(c.clear_zone.setup_visuals(ax))
        self.visual_artists.extend(c.stack_zone.setup_visuals(ax))

        # Setup Boxes
        for box in c.boxes:
            self.visual_artists.extend(box.setup_visuals(ax))

        # Setup Robots
        self.visual_artists.extend(c.husky.setup_visuals(ax))
        self.visual_artists.extend(c.anymal.setup_visuals(ax))
        
        for bot in c.puzzlebots:
            self.visual_artists.extend(bot.setup_visuals(ax))
            
        for arm in c.xarms:
            self.visual_artists.extend(arm.setup_visuals(ax))

        # ── LiDAR rays (Keep in sim.py since they are world-level) ──────────
        self.lidar_lines = [
            ax.plot([], [], '-', color=C_LIDAR, lw=0.3, zorder=2)[0]
            for _ in range(self.lidar.n_rays)
        ]

        # ── Time / Phase text ──────────────────────────────────────────────
        self.time_text  = ax.text(0.01, 0.99, '', transform=ax.transAxes,
                                   color=C_TEXT, fontsize=8, fontfamily='monospace',
                                   va='top', zorder=10)
        self.phase_text = ax.text(0.01, 0.93, '', transform=ax.transAxes,
                                   color='#f59e0b', fontsize=8, fontfamily='monospace',
                                   va='top', zorder=10)

    # ── CAMERA PANEL ─────────────────────────────────────────────────────────

    def _setup_camera_panel(self):
        ax = self.ax_cam
        ax.set_facecolor('#050a14')
        ax.axis('off')
        # Placeholder 320×180 black image
        blank = np.zeros((180, 320, 3), dtype=np.uint8)
        self.cam_imshow = ax.imshow(blank, aspect='auto',
                                    interpolation='nearest', zorder=2)
        ax.set_title('')   # title is updated dynamically each frame
        self.cam_title = ax.text(
            0.5, 1.02, 'CAMERA', transform=ax.transAxes,
            color='#64748b', fontsize=7, fontfamily='monospace',
            ha='center', va='bottom', zorder=6)
        # Overlay text for perception stats
        self.cam_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                                color='#00ff88', fontsize=6,
                                fontfamily='monospace', va='top', zorder=5)

    # ── METRICS PANEL ────────────────────────────────────────────────────────

    def _setup_metrics_panel(self):
        ax = self.ax_met
        ax.set_facecolor(C_BG)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.05, 0.96, 'HACKATHON METRICS', color='#64748b',
                fontsize=7, fontfamily='monospace', va='top', fontweight='bold')
        ax.text(0.05, 0.88, '─' * 20, color='#1e293b',
                fontsize=6, fontfamily='monospace', va='top')

        self.met_texts = {}
        rows = [
            ('cmax',     'Cmax (makespan)',  0.80),
            ('replans',  'Replans (A*)',     0.70),
            ('avoided',  'Cols avoided',     0.60),
            ('obs_seen', 'Obstacles seen',   0.50),
            ('lms_seen', 'Landmarks seen',   0.40),
            ('sep_m',    '─' * 20,           0.30),
            ('task_C',   'Box C placed',     0.22),
            ('task_B',   'Box B placed',     0.14),
            ('task_A',   'Box A placed',     0.06),
        ]
        for key, lbl, y in rows:
            if key.startswith('sep'):
                ax.text(0.05, y, lbl, color='#1e293b',
                        fontsize=6, fontfamily='monospace', va='top')
                self.met_texts[key] = ax.text(0.05, y, lbl, color='#1e293b',
                                              fontsize=6, fontfamily='monospace',
                                              va='top')
            else:
                ax.text(0.04, y, lbl + ':', color='#64748b',
                        fontsize=6.5, fontfamily='monospace', va='top')
                self.met_texts[key] = ax.text(0.62, y, '—', color=C_TEXT,
                                              fontsize=6.5, fontfamily='monospace',
                                              va='top')

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

    def _hough_overlay(self, img: np.ndarray, horizon: int) -> tuple:
        """
        Run Canny + probabilistic Hough on the floor region of *img*.
        Returns (overlaid_img, n_segments).
        """
        out = img.copy()
        gray = rgb2gray(img)

        # Restrict edge detection to floor region (below horizon)
        floor_gray = np.zeros_like(gray)
        floor_gray[horizon:] = gray[horizon:]

        edges = canny(floor_gray, sigma=1.2, low_threshold=0.08,
                      high_threshold=0.20)
        segments = probabilistic_hough_line(
            edges, threshold=12, line_length=18, line_gap=6)

        for (x0, y0), (x1, y1) in segments:
            rr, cc = draw_line(y0, x0, y1, x1)
            valid  = (rr >= 0) & (rr < out.shape[0]) & \
                     (cc >= 0) & (cc < out.shape[1])
            out[rr[valid], cc[valid]] = (0, 255, 160)   # bright green

        return out, len(segments)

    def _update_camera(self):
        """Display latest annotated camera image with Hough lane overlays."""
        c = self.coord
        img = c.last_annotated_img
        if img is None:
            return

        # Apply Hough only when the ANYmal camera is active
        if c.current_robot_name == 'ANYmal':
            display_img, n_seg = self._hough_overlay(img, c.last_cam_horizon)
            hough_info = f"  HOUGH:{n_seg}seg"
        else:
            display_img = img
            hough_info  = ""

        self.cam_imshow.set_data(display_img)

        self.cam_title.set_text(
            f'CAMERA ({c.current_robot_name} POV)  z={c.current_altitude:.1f}m')

        obs_n  = len(c.last_obstacles_det)
        lm_n   = len(c.last_landmarks_det)
        lm_ids = [f"LM{l[0]}" for l in c.last_landmarks_det]
        self.cam_text.set_text(
            f"OBS:{obs_n}  LM:{lm_n} {' '.join(lm_ids)}{hough_info}")

    def _update_metrics(self):
        """Update hackathon metrics panel."""
        c   = self.coord
        m   = c.status().get('metrics', {})
        sb  = {b.id: b.stacked for b in c.stack_boxes}

        self.met_texts['cmax'].set_text(f"{m.get('cmax', 0):.1f} s")
        self.met_texts['replans'].set_text(str(m.get('replans', 0)))
        self.met_texts['avoided'].set_text(str(m.get('collisions_avoided', 0)))
        self.met_texts['obs_seen'].set_text(str(m.get('obstacles_seen', 0)))
        self.met_texts['lms_seen'].set_text(str(m.get('landmarks_seen', 0)))

        for box_id, key in [('C', 'task_C'), ('B', 'task_B'), ('A', 'task_A')]:
            placed = sb.get(box_id, False)
            self.met_texts[key].set_text('✓' if placed else '…')
            self.met_texts[key].set_color('#10b981' if placed else '#f59e0b')

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

        # ── Update Entity Visuals ────────────────────────────────────────────
        for box in c.boxes:
            box.update_visuals()

        c.husky.update_visuals(self.ax)
        c.anymal.update_visuals(self.ax)

        for bot in c.puzzlebots:
            bot.update_visuals(self.ax)
            
        for bot in c.puzzlebots:
            bot.update_visuals(self.ax)

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
        # ── HUD ────────────────────────────────────────────────────────────
        self.time_text.set_text(f't = {c.t:.1f}s')
        self.phase_text.set_text(f'Phase: {c.phase.name}')

        # ── Camera panel ───────────────────────────────────────────────────
        self._update_camera()

        # ── Metrics panel ──────────────────────────────────────────────────
        self._update_metrics()

        # ── Info panel ─────────────────────────────────────────────────────
        self._update_info()

        return (self.visual_artists + self.lidar_lines +
                [self.time_text, self.phase_text, self.cam_imshow, self.cam_text,
                 self.cam_title, *list(self.met_texts.values()),
                 *list(self.info_texts.values())])

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
            backend = matplotlib.get_backend().lower()
            if 'agg' in backend and 'tkagg' not in backend and 'qt' not in backend:
                print(f"Non-interactive backend '{matplotlib.get_backend()}'; skipping live display.")
                print("Run with --save to write robot_sim.mp4 instead.")
                return
            self.fig.suptitle('TE3002B — Almacén Robótico Colaborativo',
                              color=C_TEXT, fontfamily='monospace', fontsize=10, y=0.98)
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
