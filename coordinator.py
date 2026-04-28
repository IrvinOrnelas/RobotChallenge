"""
coordinator.py
Multi-robot task coordinator — correct mission logic per TE3002B spec.

CORRECT MISSION FLOW:
─────────────────────────────────────────────────────────────────────────────
PHASE 1 — CLEARING (Husky only)
  · Husky detects 3 large obstacle boxes (B1,B2,B3) in corridor via LiDAR
  · Husky pushes EACH box out of the 6×2 m corridor, one at a time
  · PuzzleBots are RIDING on ANYmal's back — they do NOT move yet
  · ANYmal waits at the start zone
  · Success: all 3 obstacle boxes are outside corridor rectangle

PHASE 2 — TRANSPORTING (ANYmal carrying PuzzleBots)
  · ANYmal walks through the now-clear corridor
  · 3 PuzzleBots are ON ANYmal's back (passive cargo)
  · ANYmal navigates to pdest = (11.0, 3.6) with error < 0.15 m
  · Singularity monitoring on every leg every step (|det J| > 1e-3)
  · Success: ANYmal arrives at work zone

PHASE 3 — DEPLOYING
  · ANYmal is stationary at pdest
  · PuzzleBots are placed one by one at deploy positions (xArm action)

PHASE 4 — STACKING (PuzzleBots, ONE AT A TIME — time-slotting)
  · Stack order MANDATORY: C first (bottom), then B (middle), then A (top)
  · PuzzleBot 0 picks box C → carries it → places it at stack target
  · ONLY AFTER C is placed: PuzzleBot 1 picks box B → stacks on C
  · ONLY AFTER B is placed: PuzzleBot 2 picks box A → stacks on B
  · Each pick/place uses arm FK/IK + τ = Jᵀf force control
  · Success: A-B-C stacked in correct order, stable
─────────────────────────────────────────────────────────────────────────────
"""
from enum import Enum, auto
import numpy as np
from sklearn.cluster import KMeans
from utils import check_aabb_collision
from utils import norm2, wrap_angle, clamp
from classes.elements.box import Box
from classes.elements.zone import Zone
from utils import get_aabb_distance, detect_collision_groups, propagate_push_force, check_aabb_collision

# ── ENUMS ────────────────────────────────────────────────────────────────────

class Phase(Enum):
    STARTING     = auto()
    CLEARING     = auto()
    TRANSPORTING = auto()
    DEPLOYING    = auto()
    STACKING     = auto()
    DONE         = auto()
    ERROR        = auto()


class HuskyFSM(Enum):
    SCAN     = auto()
    APPROACH = auto()
    ALIGN    = auto()
    PUSH     = auto()
    RETREAT  = auto()
    IDLE     = auto()


class PuzzleBotFSM(Enum):
    RIDING   = auto()   # on ANYmal's back
    DEPLOYED = auto()   # placed on ground, waiting turn
    APPROACH = auto()   # driving toward target box
    GRASPING = auto()   # arm moving to grasp pose
    CARRYING = auto()   # driving to stack while holding box
    PLACING  = auto()   # arm lowering box onto stack
    DONE     = auto()


# ── COORDINATOR ───────────────────────────────────────────────────────────────

class Coordinator:
    """
    Orchestrates Husky, ANYmal and 3 PuzzleBots through the full mission.

    Key invariants:
    · PuzzleBots stay on ANYmal during CLEARING and TRANSPORTING phases.
    · Stacking is strictly sequential C→B→A (time-slotting, no parallelism).
    · ANYmal monitors |det J| every step and counts violations.
    """

    # Where each PuzzleBot is placed after ANYmal arrives (relative to pdest)
    DEPLOY_OFFSETS = [
        np.array([ 0.0,  0.5]),
        np.array([ 0.0,  0.0]),
        np.array([ 0.0, -0.5]),
    ]

    # Riding offsets on ANYmal's back (body frame x-forward, y-left)
    RIDING_OFFSETS = [
        np.array([ 0.15,  0.12]),
        np.array([ 0.00,  0.00]),
        np.array([-0.15, -0.12]),
    ]

    def __init__(self, husky, anymal, puzzlebots: list,
                 obstacle_boxes: list, stack_boxes: list,
                 clear_zone: list, stack_zone: list,
                 xarms: list = None,
                 anymal_dest=(11.0, 3.6),
                 stack_target=(12.0, 3.0),
                 lidar: object = None):
        
        self.husky      = husky
        self.anymal     = anymal
        self.puzzlebots = puzzlebots

        self.anymal_dest  = np.array(anymal_dest,  dtype=float)
        self.stack_target = np.array(stack_target, dtype=float)

        # Store separate lists for logic
        self.obstacle_boxes = obstacle_boxes
        self.stack_boxes = stack_boxes
        
        # Combined list for the sim.py renderer
        self.boxes = self.obstacle_boxes + self.stack_boxes
        
        # ── Zones ────────────────────────────────────────────────────────────
        self.clear_zone = clear_zone
        self.stack_zone = stack_zone
         
        # ── Phase ────────────────────────────────────────────────────────────
        self.phase = Phase.STARTING

        # ── Husky state ──────────────────────────────────────────────────────
        self.lidar = lidar
        self.husky_fsm     = HuskyFSM.SCAN
        self.husky_target_pt = None
        self.husky_box_idx = 0
        self.husky_origins = [(b.x, b.y) for b in self.obstacle_boxes]

        # ── ANYmal state ─────────────────────────────────────────────────────
        self.det_J_violations = 0

        # ── PuzzleBot state ───────────────────────────────────────────────────
        # All bots start RIDING on ANYmal
        self.pb_fsm          = [PuzzleBotFSM.RIDING] * 3
        self.deploy_idx      = 0
        self.active_stacker  = 0       # time-slot index (0→1→2)
        self.grasp_timer     = [0.0] * 3
        # Assignment: PB0→C (bottom), PB1→B (middle), PB2→A (top)
        self.pb_assignment   = {0: 'C', 1: 'B', 2: 'A'}
        self.stack_layer_h   = 0.35    # height added per stacked box

        self.t   = 0.0
        self.log = []
        
        # ── xArm state ────────────────────────────────────────────────────────
        self.xarms = xarms or []
        self.unassigned_pbs = [0, 1, 2]
        self.xarm_fsm = [{'state': 'IDLE', 'pb': None, 'timer': 0.0} for _ in self.xarms]

    # ── LOGGING ──────────────────────────────────────────────────────────────

    def _log(self, msg):
        entry = f"[{self.t:7.2f}s] {msg}"
        self.log.append(entry)
        print(entry)

    # ── MAIN STEP ────────────────────────────────────────────────────────────

    def step(self, dt):
        self.t += dt
        if   self.phase == Phase.STARTING:     self._do_starting()
        elif self.phase == Phase.CLEARING:     self._do_clearing(dt)
        elif self.phase == Phase.TRANSPORTING: self._do_transporting(dt)
        elif self.phase == Phase.DEPLOYING:    self._do_deploying(dt)
        elif self.phase == Phase.STACKING:     self._do_stacking(dt)

    def task_completed(self):
        return self.phase == Phase.DONE

    # ── PHASE: STARTING ──────────────────────────────────────────────────────

    def _do_starting(self):
        """Put all PuzzleBots on ANYmal and start clearing."""
        for i in range(3):
            self._snap_bot_to_anymal(i)
            self.pb_fsm[i] = PuzzleBotFSM.RIDING
        self._log("═══ PHASE 1: CLEARING — Husky pushes obstacle boxes out of corridor ═══")
        self.phase = Phase.CLEARING

    # ── PHASE: CLEARING ──────────────────────────────────────────────────────

     # ── PHASE: CLEARING ──────────────────────────────────────────────────────


    def _do_clearing(self, dt):
        """
        Husky clears corridor of B1, B2, B3 one by one.
        ANYmal stays still. PuzzleBots stay on ANYmal's back.
        """
        # Keep bots riding on stationary ANYmal
        for i in range(3):
            self._snap_bot_to_anymal(i)
        # 1. SCAN: Look for obstacles inside the Clear Zone
        if self.husky_fsm == HuskyFSM.SCAN:
            # Count exactly how many boxes still overlap with the clear zone area
            boxes_left = 0
            for b in self.obstacle_boxes:
                if check_aabb_collision(b.get_bounds(), self.clear_zone.get_bounds()):
                    boxes_left += 1
                    
            obstacles = [{'x': b.x, 'y': b.y, 'w': b.w, 'h': b.h} for b in self.boxes]
            ranges, angles = self.lidar.scan(self.husky.pose, obstacles)
            points = self.lidar.get_points(self.husky.pose, ranges, angles)
                    
            # Ask the Zone what is inside it
            points_in_zone = self.clear_zone.get_points_inside(points)
            if boxes_left == 0:
                self._log("✅ CLEARING done — corridor clear")
                self._log("═══ PHASE 2: TRANSPORTING — ANYmal walks to work zone ═══")
                self.phase = Phase.TRANSPORTING
                return

            else:
                # ML Clustering to target the remaining boxes
                if len(points_in_zone) >= 5:
                    k = max(1, boxes_left) # Prevent k=0 if there's lingering noise
                    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
                    labels = kmeans.fit_predict(points_in_zone)
                    centroids = kmeans.cluster_centers_
                    
                    # Pick the centroid that is closest to the Husky
                    hx, hy = self.husky.x, self.husky.y
                    closest_centroid = min(centroids, key=lambda c: norm2(c[0] - hx, c[1] - hy))
                    
                    self.husky_target_pt = closest_centroid
                    
                    self.husky_fsm = HuskyFSM.APPROACH
                    self._log(f" Target cluster locked at ({self.husky_target_pt[0]:.2f}, {self.husky_target_pt[1]:.2f})")
                    
            if self.husky_box_idx >= len(self.obstacle_boxes):
                self._log("✅ CLEARING done — corridor clear")
                self._log("═══ PHASE 2: TRANSPORTING — ANYmal walks to work zone ═══")
                self.phase = Phase.TRANSPORTING
                return

        # 2. APPROACH: Drive behind the detected target
        elif self.husky_fsm == HuskyFSM.APPROACH:
            target_x, target_y = self.husky_target_pt
            staging_pt = [target_x, target_y - 0.75] # Go comfortably behind the box
            
            other_boxes = [b for b in self.obstacle_boxes if b.x != target_x]
            safe_target = self._get_avoidance_target(self.husky, staging_pt, other_boxes, safe_dist=0.3)
            
            dist = norm2(staging_pt[0] - self.husky.x, staging_pt[1] - self.husky.y)
            v, w = self.husky.model.compute_velocity_command(self.husky.pose, safe_target)
            wr, wl = self.husky.model.inverse_kinematics(v, w)
            self.husky.step(dt, wr, wl)

            # Update telemetry HUD
            self.husky.v_cmd, self.husky.w_cmd = v, w

            if dist < 0.2:
                self.husky_fsm = HuskyFSM.ALIGN


        # 3. ALIGN: Face the target
        elif self.husky_fsm == HuskyFSM.ALIGN:
            target_x, target_y = self.husky_target_pt
            angle_to_target = np.arctan2(target_y - self.husky.y, target_x - self.husky.x)
            ang_err = wrap_angle(angle_to_target - self.husky.pose[2])
            w = clamp(2.0 * ang_err, -1.5, 1.5)
            wr, wl = self.husky.model.inverse_kinematics(0.0, w)
            self.husky.step(dt, wr, wl)

            # Update telemetry HUD
            self.husky.v_cmd, self.husky.w_cmd = 0.0, w

            if abs(ang_err) < 0.08:
                self.husky_fsm = HuskyFSM.PUSH

            self.push_timer = 0.0

        # 4. PUSH: Ram straight forward to shove the box
        elif self.husky_fsm == HuskyFSM.PUSH:
            self.push_timer += dt
            v, w = 0.5, 0.0 # Push forward steadily
            wr, wl = self.husky.model.inverse_kinematics(v, w)
            self.husky.step(dt, wr, wl)

            # Update telemetry HUD
            self.husky.v_cmd, self.husky.w_cmd = v, w

            husky_bounds = self.husky.get_bounds()
            target_box_cleared = True
            
            mover_vel = (v * np.cos(self.husky.theta), v * np.sin(self.husky.theta))
            obstacle_list = [b for b in self.boxes if b.obstacle_box]
            propagate_push_force(husky_bounds, mover_vel, obstacle_list, dt)

            # 2. Re-calculate bounds after movement to see if it left the zone
            for box in obstacle_list:
                if check_aabb_collision(box.get_bounds(), self.clear_zone.get_bounds()):
                    target_box_cleared = False

            # Stop pushing if Husky leaves the zone OR as a safety timeout
            if target_box_cleared or self.push_timer > 15.0:
                self.husky_fsm = HuskyFSM.RETREAT
                self.retreat_timer = 0.0


        # 5. RETREAT: Back up briefly to un-stick, then scan again
        elif self.husky_fsm == HuskyFSM.RETREAT:
            self.retreat_timer += dt
            v, w = -0.5, 0.0 # Reverse
            wr, wl = self.husky.model.inverse_kinematics(v, w)
            self.husky.step(dt, wr, wl)
            # Update telemetry HUD
            self.husky.v_cmd, self.husky.w_cmd = v, w
            # Only back up for 5.0 seconds, then immediately scan again
            if self.retreat_timer > 5.0:
                self.husky_fsm = HuskyFSM.SCAN 

    # ── PHASE: TRANSPORTING ───────────────────────────────────────────────────

    def _do_transporting(self, dt):
        """
        ANYmal walks to pdest carrying PuzzleBots on its back.
        PuzzleBots move rigidly with ANYmal.
        """
        # PuzzleBots ride on ANYmal
        for i in range(3):
            self._snap_bot_to_anymal(i)

        # Singularity monitoring every step
        for name, leg in self.anymal.legs.items():
            d = abs(leg.det_J())
            if d < 1e-3:
                self.det_J_violations += 1

        # Calculate APF steering
        safe_dest = self._get_avoidance_target(self.anymal, self.anymal_dest, self.obstacle_boxes, safe_dist=1.0)
        self.anymal.navigate_to(safe_dest, dt)
        
        true_dist = norm2(self.anymal_dest[0] - self.anymal.x, 
                          self.anymal_dest[1] - self.anymal.y)

        if true_dist < 0.15:
            # Force ANYmal to stop walking
            self.anymal.step(dt, 0.0, 0.0)
            
            self._log(f"✅ ANYmal at pdest, err={true_dist:.4f} m | "
                      f"det_J violations={self.det_J_violations}")
            self._log("═══ PHASE 3: DEPLOYING — placing PuzzleBots at work surface ═══")
            self.phase     = Phase.DEPLOYING
            self.deploy_idx = 0

    # ── PHASE: DEPLOYING ─────────────────────────────────────────────────────

    def _do_deploying(self, dt):
        """
        ANYmal is still. PuzzleBots are placed one by one at deploy positions.
        (In real life the xArm does this; here we snap them instantly.)
        """
        
        all_idle = True
        
        for pb_idx in self.unassigned_pbs:
            self._snap_bot_to_anymal(pb_idx)
            
        for i, arm in enumerate(self.xarms):
            fsm = self.xarm_fsm[i]
            
            if fsm['state'] in ['REACHING', 'GRABBING'] and fsm['pb'] is not None:
                self._snap_bot_to_anymal(fsm['pb'])
                
            if fsm['state'] == 'IDLE':
                if len(self.unassigned_pbs) > 0:
                    pb_idx = self.unassigned_pbs.pop(0)
                    fsm['pb'] = pb_idx
                    pb = self.puzzlebots[pb_idx]
                    arm.set_target(pb.x, pb.y, tz=-0.05)
                    fsm['state'] = 'REACHING'
                    all_idle = False
            
            elif fsm['state'] == 'REACHING':
                all_idle = False
                if arm.step(dt):
                    fsm['state'] = 'GRABBING'
                    fsm['timer'] = 0.
                    
            elif fsm['state'] == 'GRABBING':
                all_idle = False
                fsm['timer'] += dt
                if fsm['timer'] > 0.5:
                    pb_idx = fsm['pb']
                    deploy_pos = self.anymal_dest + self.DEPLOY_OFFSETS[pb_idx]
                    arm.set_target(deploy_pos[0], deploy_pos[1])
                    fsm['state'] = 'MOVING'
                    
            elif fsm['state'] == 'MOVING':
                all_idle = False
                arrived = arm.step(dt)
                pb_idx = fsm['pb']
                
                self.puzzlebots[pb_idx].pose[0] = arm.ee_x
                self.puzzlebots[pb_idx].pose[1] = arm.ee_y
                
                if arrived:
                    fsm['state'] = 'RELEASING'
                    fsm['timer'] = 0.0
                    
            elif fsm['state'] == 'RELEASING':
                all_idle = False
                fsm['timer'] += dt
                if fsm['timer'] > 0.3:
                    pb_idx = fsm['pb']
                    pb = self.puzzlebots[pb_idx]
                    self.pb_fsm[pb_idx] = PuzzleBotFSM.DEPLOYED
                    arm.set_target(pb.x, pb.y, tz=-0.05)
                    fsm['state'] = 'RETURNING'
                    fsm['pb'] = None
                    self._log(f"  xArm {i} desplegó el PuzzleBot {pb_idx}")
                    
            elif fsm['state'] == 'RETURNING':
                if not arm.step(dt):
                    all_idle = False
                else:
                    fsm['state'] = 'IDLE'
                    
        if all_idle and len(self.unassigned_pbs) == 0:
            self._log("✅ Los xArms terminaron de descargar los 3 PuzzleBots")
            self._log("═══ PHASE 4: STACKING — C → B → A (time-slotting) ═══")
            self.active_stacker = 0
            self.phase = Phase.STACKING

    # ── PHASE: STACKING ──────────────────────────────────────────────────────

    def _do_stacking(self, dt):
        """
        TIME-SLOTTING: only one PuzzleBot active at a time.
        Order enforced: PB0 stacks C (layer 0), then PB1 stacks B (layer 1),
        then PB2 stacks A (layer 2).
        """
        if self.active_stacker >= 3:
            self._log("✅ STACKING complete — stack C→B→A is stable. MISSION DONE.")
            self.phase = Phase.DONE
            return

        i      = self.active_stacker
        bot    = self.puzzlebots[i]
        fsm    = self.pb_fsm[i]
        box_id = self.pb_assignment[i]
        box    = next(b for b in self.stack_boxes if b.id == box_id)
        layer  = i   # 0=C(bottom), 1=B(middle), 2=A(top)

        # ── Activate idle bot ────────────────────────────────────────────────
        if fsm == PuzzleBotFSM.DEPLOYED:
            self.pb_fsm[i] = PuzzleBotFSM.APPROACH
            self._log(f"  [Slot {i}] PuzzleBot {i} → picking box {box_id}")
            return

        # ── APPROACH: drive toward the box ───────────────────────────────────
        if fsm == PuzzleBotFSM.APPROACH:
            # Obstacles are the other PuzzleBots
            other_bots = [b for idx, b in enumerate(self.puzzlebots) if idx != i]
            safe_target = self._get_avoidance_target(bot, [box.x, box.y], other_bots, safe_dist=0.4)
            
            arrived = bot.navigate_to(safe_target, dt)
            bot.step(dt)
            if arrived:
                self._log(f"  [Slot {i}] PuzzleBot {i} reached box {box_id}")
                self.pb_fsm[i]    = PuzzleBotFSM.GRASPING
                self.grasp_timer[i] = 0.0

        # ── GRASPING: arm descends and grips ─────────────────────────────────
        elif fsm == PuzzleBotFSM.GRASPING:
            bot.arm.set_ee_target(0.12, 0.0, 0.10)
            bot.step(dt)
            self.grasp_timer[i] += dt

            # Force control: τ = Jᵀ · f_grip
            f_grip = np.array([0.0, 0.0, -5.0])
            tau    = bot.arm.force_to_torque(f_grip)

            if self.grasp_timer[i] > 0.6:
                box.carried_by = i
                self._log(f"  [Slot {i}] Grasped {box_id} | τ={np.round(tau,3)}")
                self.pb_fsm[i]    = PuzzleBotFSM.CARRYING
                self.grasp_timer[i] = 0.0

        # ── CARRYING: navigate to stack zone ─────────────────────────────────
        elif fsm == PuzzleBotFSM.CARRYING:
            other_bots = [b for idx, b in enumerate(self.puzzlebots) if idx != i]
            safe_target = self._get_avoidance_target(bot, self.stack_target, other_bots, safe_dist=0.4)
            
            arrived = bot.navigate_to(safe_target, dt)
            bot.step(dt)
            # Box follows arm EE in world frame
            ee = bot.arm.ee_position()
            box.x = bot.x + ee[0]*np.cos(bot.theta) - ee[1]*np.sin(bot.theta)
            box.y = bot.y + ee[0]*np.sin(bot.theta) + ee[1]*np.cos(bot.theta)

            if arrived:
                self._log(f"  [Slot {i}] PuzzleBot {i} at stack zone — placing {box_id}")
                self.pb_fsm[i]    = PuzzleBotFSM.PLACING
                self.grasp_timer[i] = 0.0

        # ── PLACING: lower onto stack with force control ──────────────────────
        elif fsm == PuzzleBotFSM.PLACING:
            z_rest = layer * self.stack_layer_h + 0.04
            bot.arm.set_ee_target(0.12, 0.0, max(0.04, z_rest))
            bot.step(dt)
            self.grasp_timer[i] += dt

            f_place = np.array([0.0, 0.0, -2.0])
            tau     = bot.arm.force_to_torque(f_place)

            if self.grasp_timer[i] > 0.8:
                # Finalise box position
                box.x          = self.stack_target[0]
                box.y          = self.stack_target[1]
                box.stack_layer = layer
                box.stacked    = True
                box.carried_by = None
                bot.arm.home()
                self.pb_fsm[i] = PuzzleBotFSM.DONE
                self._log(f"  [Slot {i}] ✓ Box {box_id} placed at layer {layer} | "
                          f"τ={np.round(tau,3)}")
                # Unlock next time-slot
                self.active_stacker += 1
                if self.active_stacker < 3:
                    nxt = self.pb_assignment[self.active_stacker]
                    self._log(f"  ── Slot released → PuzzleBot {self.active_stacker} "
                              f"now active (box {nxt}) ──")

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _snap_bot_to_anymal(self, i):
        """Rigidly attach PuzzleBot i to ANYmal's back."""
        off = self.RIDING_OFFSETS[i]
        th  = self.anymal.pose[2]
        self.puzzlebots[i].pose[0] = (self.anymal.x
                                      + off[0]*np.cos(th) - off[1]*np.sin(th))
        self.puzzlebots[i].pose[1] = (self.anymal.y
                                      + off[0]*np.sin(th) + off[1]*np.cos(th))
        self.puzzlebots[i].pose[2] = th
        
    def _get_avoidance_target(self, robot, true_target, obstacles, safe_dist=0.7, k_rep=0.4):
        """Calculates a virtual target using Artificial Potential Fields."""
        rx, ry = robot.x, robot.y
        tx, ty = true_target[0], true_target[1]
        
        # 1. Attractive vector towards the true target
        dx, dy = tx - rx, ty - ry
        dist_t = norm2(dx, dy)
        
        if dist_t > 0:
            scale = min(1.0, dist_t)
            dx = (dx / dist_t) * scale
            dy = (dy / dist_t) * scale
            
            # 2. Repulsive vectors away from obstacles
            rep_x, rep_y = 0.0, 0.0
            for obs in obstacles:
                ox, oy = obs.x, obs.y
                dist_o = norm2(rx - ox, ry - oy)
                if 0.01 < dist_o < safe_dist:
                    mag = k_rep * (1.0 / dist_o - 1.0 / safe_dist) / (dist_o**2)
                    rep_x += mag * (rx - ox) / dist_o
                    rep_y += mag * (ry - oy) / dist_o

        # 3. Combine vectors to create a new virtual waypoint
        return [rx + dx + rep_x, ry + dy + rep_y] 

    # ── STATUS ────────────────────────────────────────────────────────────────

    def status(self):
        return {
            'phase'  : self.phase.name,
            't'      : round(self.t, 2),
            'husky'  : {'pos': self.husky.pose[:2].tolist(),
                        'v_cmd': round(self.husky.v_cmd, 4),
                        'v_meas': round(self.husky.v_meas, 4)},
            'anymal' : {'pos': self.anymal.pose[:2].tolist(),
                        'dist_goal': round(norm2(
                            self.anymal_dest[0]-self.anymal.x,
                            self.anymal_dest[1]-self.anymal.y), 4),
                        'det_J_violations': self.det_J_violations},
            'puzzlebots': [{'id': i, 'state': self.pb_fsm[i].name,
                            'pos': self.puzzlebots[i].pose[:2].tolist(),
                            'box': self.pb_assignment.get(i)}
                           for i in range(3)],
            'stack_boxes': [{'id': b.id, 'stacked': b.stacked,
                             'layer': b.stack_layer}
                            for b in self.stack_boxes],
        }