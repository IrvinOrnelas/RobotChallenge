import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'classes', 'husky'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation

from classes.husky.husky import Husky
from classes.husky.lidar import LiDAR

ZONE_CENTER = np.array([4.0, 0.0])
ZONE_RADIUS = 1.8

BOX_W, BOX_H = 0.5, 0.5
box = {
    'x': ZONE_CENTER[0] - BOX_W / 2,
    'y': ZONE_CENTER[1] - BOX_H / 2,
    'w': BOX_W, 'h': BOX_H,
}

PUSH_DIR = np.array([1.0, 0.0])

HUSKY_L, HUSKY_W = 0.990, 0.670

ROBOT_RADIUS = 0.42 

def resolve_robot_box(pos, push_mode):
    cx, cy   = pos
    bx0, bx1 = box['x'], box['x'] + box['w']
    by0, by1 = box['y'], box['y'] + box['h']

    near_x = np.clip(cx, bx0, bx1)
    near_y = np.clip(cy, by0, by1)
    dx, dy  = cx - near_x, cy - near_y
    dist    = np.hypot(dx, dy)

    if dist >= ROBOT_RADIUS or dist < 1e-9:
        return pos   # sin penetración

    overlap = ROBOT_RADIUS - dist
    sep_x   = dx / dist * overlap
    sep_y   = dy / dist * overlap

    to_bc       = box_center() - pos
    from_behind = np.dot(to_bc, PUSH_DIR) > 0

    if push_mode and from_behind:
        box['x'] -= sep_x
        box['y'] -= sep_y
        return pos
    else:
        # Robot rebota
        return np.array([cx + sep_x, cy + sep_y])


ARRIVAL_TOL   = 0.15  
BEHIND_OFFSET = 0.90  
DT            = 0.05 

robot = Husky(pose=(0.5, -1.8, 0.0))
robot.model.maxspeed = 6          

lidar = LiDAR(n_rays=360, max_range=8.0, fov=2 * np.pi, noise_std=0.01)
robot.attach_lidar(lidar)

state        = 'SCAN'
box_centroid = None
behind_pt    = None
trajectory   = []

t_data     = []          
vlin_data  = []          
vang_data  = []          
rx_data    = []          
ry_data    = []         
dzone_data = []          

def box_center():
    return np.array([box['x'] + BOX_W / 2, box['y'] + BOX_H / 2])

def box_in_zone():
    return np.linalg.norm(box_center() - ZONE_CENTER) < ZONE_RADIUS


fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-3.5, 3.0)
ax.set_aspect('equal')
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')
ax.tick_params(colors='#444')
for sp in ax.spines.values():
    sp.set_color('#333')

zone_patch = plt.Circle(ZONE_CENTER, ZONE_RADIUS,
                        facecolor='#7b1fa2', alpha=0.15,
                        edgecolor='#ce93d8', linewidth=1.5,
                        linestyle='--', zorder=1)
ax.add_patch(zone_patch)
ax.text(ZONE_CENTER[0], ZONE_CENTER[1] + ZONE_RADIUS + 0.15,
        'ZONA PELIGROSA', ha='center', color='#ce93d8', fontsize=8,
        fontweight='bold')

traj_line, = ax.plot([], [], '-', color='#3498db', alpha=0.28, lw=1.2, zorder=1)

lidar_sc = ax.scatter([], [], s=3, c='#f39c12', alpha=0.50, zorder=2)

patch_box = patches.Rectangle(
    (box['x'], box['y']), BOX_W, BOX_H,
    facecolor='#e67e22', edgecolor='#f0b27a', linewidth=1.5, zorder=3
)
ax.add_patch(patch_box)

patch_box_col = patches.Rectangle(
    (box['x'], box['y']), BOX_W, BOX_H,
    fill=False, edgecolor='white', linewidth=0.8,
    linestyle=':', alpha=0.45, zorder=4
)
ax.add_patch(patch_box_col)

patch_robot = patches.FancyBboxPatch(
    (-HUSKY_L / 2, -HUSKY_W / 2), HUSKY_L, HUSKY_W,
    boxstyle='round,pad=0.04',
    facecolor='#2980b9', edgecolor='#85c1e9', linewidth=1.5, zorder=5
)
ax.add_patch(patch_robot)

coll_circle = plt.Circle(
    robot.pose[:2], ROBOT_RADIUS,
    fill=False, edgecolor='#e74c3c', linewidth=1.0,
    linestyle='--', alpha=0.70, zorder=6
)
ax.add_patch(coll_circle)

head_line, = ax.plot([], [], '-', color='white', lw=2.0, zorder=7)

centroid_dot, = ax.plot([], [], 'o', color='#00bcd4', markersize=10, zorder=8,
                        label='Centroide (LiDAR)')
behind_dot,   = ax.plot([], [], 's', color='#ffeb3b', markersize=8,  zorder=8,
                        label='Punto de aproximación')

state_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                     fontsize=12, color='white', fontweight='bold',
                     va='top', fontfamily='monospace')
info_text  = ax.text(0.02, 0.89, '', transform=ax.transAxes,
                     fontsize=9, color='#aaaaaa',
                     va='top', fontfamily='monospace')

ax.legend(loc='lower right', facecolor='#161b22', edgecolor='#333',
          labelcolor='white', fontsize=9)
ax.set_title('Husky A200  |  LiDAR + Colisión + Empuje fuera de zona',
             color='white', fontsize=13, pad=10)

_DYN_BG   = '#0d1117'
_DYN_GRID = '#1e2530'
fig2, axes2 = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
fig2.patch.set_facecolor(_DYN_BG)
fig2.suptitle('Dinámica del sistema', color='white', fontsize=13, fontweight='bold')

_dyn_cfg = [
    ('Vel. lineal  v  (m/s)',   '#3498db'),
    ('Vel. angular w  (rad/s)', '#e67e22'),
    ('Posición robot  x, y (m)', '#2ecc71'),
    ('Dist. caja → zona (m)',   '#e74c3c'),
]
for i, (title, _) in enumerate(_dyn_cfg):
    a = axes2[i]
    a.set_facecolor(_DYN_BG)
    a.set_title(title, color='#aaaaaa', fontsize=9, loc='left', pad=3)
    a.tick_params(colors='#555')
    a.yaxis.label.set_color('#aaaaaa')
    for sp in a.spines.values():
        sp.set_color('#333')
    a.grid(True, color=_DYN_GRID, linewidth=0.6)

axes2[-1].set_xlabel('Tiempo (s)', color='#aaaaaa')

dyn_vlin,  = axes2[0].plot([], [], color='#3498db',  lw=1.4)
dyn_vang,  = axes2[1].plot([], [], color='#e67e22',  lw=1.4)
dyn_rx,    = axes2[2].plot([], [], color='#2ecc71',  lw=1.4, label='x')
dyn_ry,    = axes2[2].plot([], [], color='#9b59b6',  lw=1.4, label='y')
axes2[2].legend(facecolor='#161b22', edgecolor='#333',
                labelcolor='white', fontsize=8, loc='upper left')
dyn_dzone, = axes2[3].plot([], [], color='#e74c3c',  lw=1.4)
axes2[3].axhline(ZONE_RADIUS, color='#ce93d8', lw=0.9,
                 linestyle='--', label=f'límite={ZONE_RADIUS} m')
axes2[3].legend(facecolor='#161b22', edgecolor='#333',
                labelcolor='white', fontsize=8, loc='upper right')

fig2.tight_layout(rect=[0, 0, 1, 0.96])

STATE_LABELS = {
    'SCAN':      'SCAN       – escaneando con LiDAR...',
    'GO_BEHIND': 'GO_BEHIND  – navegando al punto de aproximacion',
    'PUSH':      'PUSH       – empujando (colision activa)',
    'DONE':      'DONE       – caja fuera de la zona',
}

def update(_frame):
    global state, box_centroid, behind_pt

    obstacles = [box]
    v, w = 0.0, 0.0         

    if state == 'SCAN':
        hits = robot.detect_boxes(obstacles)
        if hits:
            box_centroid = np.mean(hits, axis=0)
            behind_pt    = box_centroid - PUSH_DIR * BEHIND_OFFSET
            state        = 'GO_BEHIND'

    elif state == 'GO_BEHIND':
        if np.linalg.norm(behind_pt - robot.pose[:2]) < ARRIVAL_TOL:
            state = 'PUSH'
        else:
            v, w = robot.model.compute_velocity_command(robot.pose, behind_pt)
            robot.send_velocity(v, w)
            robot.step(DT)
            robot.pose[:2] = resolve_robot_box(robot.pose[:2], push_mode=False)

    elif state == 'PUSH':
        bc   = box_center()
        v, w = robot.model.compute_velocity_command(robot.pose, bc)
        robot.send_velocity(v, w)
        robot.step(DT)
        robot.pose[:2] = resolve_robot_box(robot.pose[:2], push_mode=True)

        if not box_in_zone():
            state = 'DONE'
            robot.send_velocity(0.0, 0.0)

    elif state == 'DONE':
        robot.send_velocity(0.0, 0.0)

    x, y, theta = robot.pose
    trajectory.append((x, y))

    if len(trajectory) > 1:
        tx, ty = zip(*trajectory)
        traj_line.set_data(tx, ty)

    ranges, angles = lidar.scan(obstacles)
    sx, sy, _      = lidar._get_sensor_pose()
    mask  = ranges < lidar.max_range - 0.1
    hit_x = sx + ranges[mask] * np.cos(angles[mask])
    hit_y = sy + ranges[mask] * np.sin(angles[mask])
    lidar_sc.set_offsets(np.c_[hit_x, hit_y] if hit_x.size else np.empty((0, 2)))

    tf = Affine2D().rotate(theta).translate(x, y) + ax.transData
    patch_robot.set_transform(tf)

    coll_circle.set_center((x, y))

    head_line.set_data([x, x + 0.55 * np.cos(theta)],
                       [y, y + 0.55 * np.sin(theta)])

    patch_box.set_xy((box['x'], box['y']))
    patch_box_col.set_xy((box['x'], box['y']))

    if not box_in_zone():
        patch_box.set_facecolor('#2ecc71')
        patch_box.set_edgecolor('#a9dfbf')

    if box_centroid is not None:
        centroid_dot.set_data([box_centroid[0]], [box_centroid[1]])
    if behind_pt is not None:
        behind_dot.set_data([behind_pt[0]], [behind_pt[1]])

    state_text.set_text(STATE_LABELS.get(state, state))
    bc     = box_center()
    d_zone = np.linalg.norm(bc - ZONE_CENTER)
    info_text.set_text(
        f'Robot  x={x:+.2f}  y={y:+.2f}  theta={np.degrees(theta):+.1f} deg\n'
        f'Caja   dist_zona={d_zone:.2f} m  (limite={ZONE_RADIUS:.1f} m)'
    )

    t_now = len(t_data) * DT
    t_data.append(t_now)
    vlin_data.append(v)
    vang_data.append(w)
    rx_data.append(x)
    ry_data.append(y)
    dzone_data.append(d_zone)

    dyn_vlin.set_data(t_data, vlin_data)
    dyn_vang.set_data(t_data, vang_data)
    dyn_rx.set_data(t_data, rx_data)
    dyn_ry.set_data(t_data, ry_data)
    dyn_dzone.set_data(t_data, dzone_data)

    for a in axes2:
        a.relim()
        a.autoscale_view()

    fig2.canvas.draw_idle()

    return (traj_line, lidar_sc, patch_robot, coll_circle, patch_box,
            patch_box_col, head_line, centroid_dot, behind_dot,
            state_text, info_text)

ani = animation.FuncAnimation(
    fig, update,
    frames=2000, interval=50, blit=False
)

plt.tight_layout()
plt.show()
