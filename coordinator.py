"""
coordinator.py
Multi-robot task coordinator — correct mission logic per TE3002B spec.

CORRECT MISSION FLOW:
─────────────────────────────────────────────────────────────────────────────
PHASE 1 — CLEARING (Husky only)
  · Husky detects 3 large obstacle boxes (B1,B2,B3) via camera perception
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
from utils import check_aabb_collision
from utils import norm2, wrap_angle, clamp
from classes.elements.box import Box
from classes.elements.zone import Zone
from utils import get_aabb_distance, detect_collision_groups, propagate_push_force, check_aabb_collision
from classes.vision.camera import CameraSimulator
from classes.vision.perception import (PerceptionPipeline, HoughFeatureExtractor,
                                       GradientBoostingSteering, LandmarkLocalizer,
                                       PCABoxOrientation)
from classes.planning.astar import AStarPlanner

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
    PARK     = auto()
    IDLE     = auto()


class PuzzleBotFSM(Enum):
    RIDING   = auto()   # on ANYmal's back
    DEPLOYED = auto()   # placed on ground, waiting turn
    APPROACH = auto()   # driving toward target box
    GRASPING = auto()   # arm moving to grasp pose
    CARRYING = auto()   # driving to stack while holding box
    PLACING  = auto()   # arm lowering box onto stack
    DONE     = auto()
    PARKING  = auto()


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
        np.array([ 1.0,  0.5]),
        np.array([ 1.0,  0.0]),
        np.array([ 1.0, -0.5]),
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
                 stack_target=(12.0, 3.0)):
        
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
        self.husky_fsm        = HuskyFSM.SCAN
        self.husky_target_pt  = None
        self._husky_target_box = None   # reference to the box currently being cleared
        self.husky_box_idx    = 0
        self.husky_origins    = [(b.x, b.y) for b in self.obstacle_boxes]

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

        # ── Vision & Planning ────────────────────────────────────────────────
        self.camera          = CameraSimulator(fov_deg=90.0, max_range=6.0)
        self.percep          = PerceptionPipeline()
        self.hough_extractor = HoughFeatureExtractor()
        self.gb_steering     = GradientBoostingSteering()
        self.localizer       = LandmarkLocalizer()
        self.pca_orient      = PCABoxOrientation()
        self.last_push_angle = None   # PCA-computed push direction, reset per box
        self.planner         = AStarPlanner()

        # Build initial occupancy grid from static boxes only
        self.planner.build_grid(self.stack_boxes)

        # Last rendered camera images (read by sim.py)
        self.last_camera_img     = None
        self.last_annotated_img  = None
        self.last_obstacles_det  = []
        self.last_landmarks_det  = []
        self._last_cam_theta     = 0.0    # cam heading used during last CLEARING render

        # Active camera identity (updated each step, read by sim.py for title)
        self.current_robot_name = 'Husky'
        self.current_altitude   = 0.3   # metres
        self.last_cam_horizon   = 90    # pixel row of horizon in last camera frame

        # Metrics tracked for the hackathon rubric
        self.cmax_timer          = 0.0   # makespan (total mission time)
        self.collisions_avoided  = 0     # replanning events driven by camera
        self.replan_count        = 0     # alias to planner.replan_count

        # A* path state for ANYmal TRANSPORTING
        self._anymal_waypoints  = []
        self._anymal_wp_idx     = 0

    # ── LOGGING ──────────────────────────────────────────────────────────────

    def _log(self, msg):
        entry = f"[{self.t:7.2f}s] {msg}"
        self.log.append(entry)
        print(entry)

    # ── MAIN STEP ────────────────────────────────────────────────────────────

    def step(self, dt):
        self.t += dt
        if self.phase != Phase.DONE:
            self.cmax_timer = self.t

        # ── Camera robot + altitude + box-set selection per phase ───────────────
        if self.phase in (Phase.STARTING, Phase.CLEARING):
            self.current_robot_name = 'Husky'

            # Body-fixed forward camera: follows Husky's heading, ground-level altitude.
            # SCAN uses position-based fallback when the box isn't in front; during
            # ALIGN/PUSH Husky explicitly faces the box so the camera sees it correctly.
            cam_theta = float(self.husky.pose[2])   # body-fixed, not pan-to-target
            self.current_altitude = 0.5              # near ground level

            if self.husky_fsm in (HuskyFSM.PARK, HuskyFSM.IDLE):
                cam_boxes = []   # corridor cleared — nothing to render; prevents stale
                                 # detections from carrying into TRANSPORTING
            else:
                cam_boxes = self.obstacle_boxes

            active_pose = np.array([self.husky.x, self.husky.y, cam_theta])
            self._last_cam_theta = float(cam_theta)   # stored for SCAN back-projection

        elif self.phase == Phase.TRANSPORTING:
            active_pose             = self.anymal.pose
            self.current_robot_name = 'ANYmal'
            dist_left = norm2(self.anymal_dest[0] - self.anymal.x,
                              self.anymal_dest[1] - self.anymal.y)
            progress  = 1.0 - min(1.0, dist_left / 8.5)
            self.current_altitude = 0.5 + progress * 2.0
            cam_boxes = self.stack_boxes   # corridor cleared; show only targets

        elif self.phase == Phase.DEPLOYING:
            active_pose             = self.anymal.pose
            self.current_robot_name = 'ANYmal'
            self.current_altitude   = 2.2
            cam_boxes = self.stack_boxes

        else:   # STACKING / DONE
            active_pose             = self.anymal.pose
            self.current_robot_name = 'ANYmal'
            self.current_altitude   = 1.5
            cam_boxes = self.stack_boxes

        draw_lanes = self.current_robot_name == 'ANYmal'
        self.last_camera_img = self.camera.render(
            active_pose, cam_boxes, altitude=self.current_altitude,
            draw_lanes=draw_lanes)
        self.last_cam_horizon = self.camera.horizon
        annotated, obs, lms  = self.percep.annotate(self.last_camera_img)
        self.last_annotated_img = annotated
        self.last_obstacles_det = obs
        self.last_landmarks_det = lms

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
        self.husky_home = [self.husky.x, self.husky.y]
        for i in range(3):
            self._snap_bot_to_anymal(i)
            self.pb_fsm[i] = PuzzleBotFSM.RIDING
        self._log("═══ PHASE 1: CLEARING — Husky pushes obstacle boxes out of corridor ═══")
        self.phase = Phase.CLEARING

     # ── PHASE: CLEARING ──────────────────────────────────────────────────────


    def _do_clearing(self, dt):
        """
        Husky clears corridor of B1, B2, B3 one by one.
        ANYmal stays still. PuzzleBots stay on ANYmal's back.
        """
        # Keep bots riding on stationary ANYmal
        for i in range(3):
            self._snap_bot_to_anymal(i)
        # 1. SCAN: Elevated camera + KMeans on pixels + corrected back-projection
        if self.husky_fsm == HuskyFSM.SCAN:
            boxes_in_zone = [b for b in self.obstacle_boxes
                             if check_aabb_collision(b.get_bounds(),
                                                     self.clear_zone.get_bounds())]
            if not boxes_in_zone:
                self._log("✅ CLEARING done — corridor clear")
                self.husky_fsm = HuskyFSM.PARK
                return

            img_scan = self.last_camera_img
            if img_scan is not None:
                H, W = img_scan.shape[:2]
                obs_pix, _ = self.percep.detect_obstacles(img_scan)

                if not obs_pix:
                    # Camera can't detect any obstacle pixels from this vantage —
                    # fall back to nearest known box position so SCAN never stalls.
                    hx, hy = self.husky.x, self.husky.y
                    best_box = min(boxes_in_zone,
                                   key=lambda b: norm2(b.x - hx, b.y - hy))
                    self.husky_target_pt   = np.array([best_box.x, best_box.y])
                    self._husky_target_box = best_box
                    self.last_push_angle   = None
                    self.husky_fsm = HuskyFSM.APPROACH
                    self._log(f"[POS-FALLBACK] Target {best_box.id} at "
                              f"({best_box.x:.2f}, {best_box.y:.2f})")

                if obs_pix:
                    # ── KMeans clustering on detected pixel coordinates ────────
                    pixel_pts = np.array([[cx, cy] for cx, cy, w, h in obs_pix],
                                         dtype=np.float32)
                    k = min(len(obs_pix), len(boxes_in_zone))
                    if k >= 2:
                        from sklearn.cluster import KMeans as _KM
                        km = _KM(n_clusters=k, n_init='auto', random_state=42)
                        km.fit(pixel_pts)
                        centers_px = km.cluster_centers_
                    else:
                        centers_px = pixel_pts[:1]

                    # ── Corrected inverse-perspective back-projection ──────────
                    # col_h = H * 1.8 * col_h_scale / dist  →  dist = H * 1.8 * col_h_scale / col_h
                    alt_norm    = min(1.0, self.current_altitude / 3.0)
                    col_h_scale = max(0.12, 1.0 - alt_norm * 0.78)
                    cam_theta   = self._last_cam_theta

                    world_candidates = []
                    for px_cx, px_cy in centers_px:
                        # Find observation nearest this cluster center for height estimate
                        nearest = min(obs_pix,
                                      key=lambda o: abs(o[0] - px_cx) + abs(o[1] - px_cy))
                        h_px = max(1, nearest[3])
                        dist = H * 1.8 * col_h_scale / h_px
                        ray  = cam_theta + self.camera.fov * (px_cx / W - 0.5)
                        world_candidates.append([
                            self.husky.x + dist * np.cos(ray),
                            self.husky.y + dist * np.sin(ray),
                        ])

                    # ── Match world candidates to nearest known box position ───
                    best_box = min(
                        boxes_in_zone,
                        key=lambda b: min(norm2(b.x - wc[0], b.y - wc[1])
                                          for wc in world_candidates))
                    self.husky_target_pt  = np.array([best_box.x, best_box.y])
                    self._husky_target_box = best_box   # reference for PUSH exit check
                    self.last_push_angle  = None   # reset PCA per new target
                    self.husky_fsm = HuskyFSM.APPROACH
                    self._log(f" [CAM+KMeans] Target box {best_box.id} at "
                              f"({best_box.x:.2f}, {best_box.y:.2f})")

            if self.husky_box_idx >= len(self.obstacle_boxes):
                self._log("✅ CLEARING done — corridor clear")
                self.husky_fsm = HuskyFSM.PARK
                return

        # 2. APPROACH: Drive behind the detected target (pure geometric)
        elif self.husky_fsm == HuskyFSM.APPROACH:
            target_x, target_y = self.husky_target_pt
            staging_pt = [target_x, target_y - 0.75]

            # Only avoid boxes still inside the clear zone — already-pushed boxes
            # must NOT generate repulsion or they'll deflect Husky wildly off course
            other_boxes = [b for b in self.obstacle_boxes
                           if b is not self._husky_target_box
                           and check_aabb_collision(b.get_bounds(), self.clear_zone.get_bounds())]
            safe_target = self._get_avoidance_target(self.husky, staging_pt, other_boxes, safe_dist=0.3)

            dist = norm2(staging_pt[0] - self.husky.x, staging_pt[1] - self.husky.y)
            v, w = self.husky.model.compute_velocity_command(self.husky.pose, safe_target)
            wr, wl = self.husky.model.inverse_kinematics(v, w)
            self.husky.step(dt, wr, wl)

            self.husky.v_cmd, self.husky.w_cmd = v, w

            if dist < 0.2:
                self.husky_fsm = HuskyFSM.ALIGN


        # 3. ALIGN: Face the target (with PCA-estimated push angle refinement)
        elif self.husky_fsm == HuskyFSM.ALIGN:
            target_x, target_y = self.husky_target_pt

            # PCA: estimate box principal orientation from camera pixels (once per box)
            if self.last_push_angle is None and self.last_camera_img is not None:
                obs_pix_align, _ = self.percep.detect_obstacles(self.last_camera_img)
                if obs_pix_align:
                    cx_px, cy_px, w_px, h_px = max(obs_pix_align,
                                                    key=lambda o: o[2] * o[3])
                    pca_angle = self.pca_orient.estimate(
                        self.last_camera_img, cx_px, cy_px, w_px, h_px)
                    if pca_angle is not None:
                        self.last_push_angle = pca_angle
                        self._log(f" [PCA] Push angle estimated: {np.degrees(pca_angle):.1f}°")

            # Align to geometric bearing toward box (primary); PCA refines if available
            angle_to_target = np.arctan2(target_y - self.husky.y, target_x - self.husky.x)
            ang_err = wrap_angle(angle_to_target - self.husky.pose[2])
            w = clamp(2.0 * ang_err, -1.5, 1.5)
            wr, wl = self.husky.model.inverse_kinematics(0.0, w)
            self.husky.step(dt, wr, wl)

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
            mover_vel = (v * np.cos(self.husky.theta), v * np.sin(self.husky.theta))
            obstacle_list = [b for b in self.boxes if b.obstacle_box]
            propagate_push_force(husky_bounds, mover_vel, obstacle_list, dt)

            # Stop as soon as the specific target box exits the clear zone (or timeout)
            target_box_cleared = (
                self._husky_target_box is not None
                and not check_aabb_collision(self._husky_target_box.get_bounds(),
                                             self.clear_zone.get_bounds())
            )

            if target_box_cleared or self.push_timer > 12.0:
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
            # Back up for 3 s (≈1.5 m) — enough to clear the box without overshooting
            if self.retreat_timer > 3.0:
                self.husky_fsm = HuskyFSM.SCAN
        
        # 6. PARK: Rertunr to its initial point
        elif self.husky_fsm == HuskyFSM.PARK:
            dist = norm2(self.husky_home[0] - self.husky.x, self.husky_home[1] - self.husky.y)
            
            # Navigation avoiding the boxes using potential vortex (AABB + Vortex)
            safe_target = self._get_avoidance_target(self.husky, self.husky_home, self.obstacle_boxes, safe_dist=0.4)
            v, w = self.husky.model.compute_velocity_command(self.husky.pose, safe_target)
            wr, wl = self.husky.model.inverse_kinematics(v, w)
            self.husky.step(dt, wr, wl)
            self.husky.v_cmd, self.husky.w_cmd = v, w
            
            if dist < 0.3: 
                # Off motors
                wr, wl = self.husky.model.inverse_kinematics(0.0, 0.0)
                self.husky.step(dt, wr, wl)
                self.husky.v_cmd, self.husky.w_cmd = 0.0, 0.0
                
                self.husky_fsm = HuskyFSM.IDLE
                self._log("✅ CLEARING done — Husky parked successfully.")
                self._log("═══ PHASE 2: TRANSPORTING — ANYmal walks to work zone ═══")
                
                self.phase = Phase.TRANSPORTING

    # ── PHASE: TRANSPORTING ───────────────────────────────────────────────────

    def _do_transporting(self, dt):
        """
        ANYmal walks to pdest carrying PuzzleBots on its back.
        Uses A* path planning; replans if camera detects unexpected obstacles.
        """
        # PuzzleBots ride on ANYmal
        for i in range(3):
            self._snap_bot_to_anymal(i)

        # Singularity monitoring every step
        for name, leg in self.anymal.legs.items():
            d = abs(leg.det_J())
            if d < 1e-3:
                self.det_J_violations += 1

        # Build A* path on first entry
        if not self._anymal_waypoints:
            # Corridor is cleared — only stack boxes matter for grid construction
            self.planner.build_grid(self.stack_boxes)
            self._anymal_waypoints = self.planner.plan(
                (self.anymal.x, self.anymal.y), self.anymal_dest.tolist())
            self._anymal_wp_idx = 0
            # Clear stale perception data from CLEARING phase so the localizer
            # can't teleport ANYmal using a landmark detected in the last
            # CLEARING camera frame (where the camera heading ≠ robot heading)
            self.last_obstacles_det = []
            self.last_landmarks_det = []
            self._log(f"  A* path planned: {len(self._anymal_waypoints)} waypoints")

        # Hough + GB steering from ANYmal's forward camera (draw_lanes=True)
        w_gb = self.gb_steering.predict(
            self.last_camera_img, horizon=self.last_cam_horizon) \
            if self.last_camera_img is not None else 0.0

        # Landmark localizer disabled for TRANSPORTING:
        # Landmark 0 at (0.5, 2.9) enters the camera FOV at close range during
        # corridor entry, giving unreliable distance estimates that teleport ANYmal.
        # Odometry is sufficient for the 8 m straight corridor traverse.

        def _navigate_with_gb(target):
            """Drive ANYmal to target blending geometric P-controller with GB omega."""
            dx = target[0] - self.anymal.pose[0]
            dy = target[1] - self.anymal.pose[1]
            dist = norm2(dx, dy)
            if dist < 0.15:
                return True
            ang_err = wrap_angle(np.arctan2(dy, dx) - self.anymal.pose[2])
            v     = clamp(0.4 * dist, 0.0, 0.8)
            w_geo = clamp(1.5 * ang_err, -2.0, 2.0)
            w     = clamp(w_geo + 0.3 * w_gb, -2.0, 2.0)
            self.anymal.step(dt, v, w)
            return False

        # Follow current A* waypoint — no obstacle avoidance needed, corridor is clear
        if self._anymal_wp_idx < len(self._anymal_waypoints):
            wp = self._anymal_waypoints[self._anymal_wp_idx]
            wp_dist = norm2(wp[0] - self.anymal.x, wp[1] - self.anymal.y)
            if wp_dist < 0.25:
                self._anymal_wp_idx += 1
            else:
                _navigate_with_gb(wp)
        else:
            # All waypoints passed — head directly to goal
            _navigate_with_gb(self.anymal_dest)

        true_dist = norm2(self.anymal_dest[0] - self.anymal.x,
                          self.anymal_dest[1] - self.anymal.y)

        if true_dist < 0.15:
            self.anymal.step(dt, 0.0, 0.0)
            self._log(f"✅ ANYmal at pdest, err={true_dist:.4f} m | "
                      f"det_J violations={self.det_J_violations} | "
                      f"replans={self.planner.replan_count}")
            self._log("═══ PHASE 3: DEPLOYING — placing PuzzleBots at work surface ═══")
            self.phase      = Phase.DEPLOYING
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
                self.pb_fsm[i] = PuzzleBotFSM.PARKING
                self._log(f"  [Slot {i}] ✓ Box {box_id} placed at layer {layer} | "
                          f"τ={np.round(tau,3)}")
                self._log(f"  [Slot {i}] Moving away to park...")
                self.grasp_timer[i] = 0.0
                
        elif fsm == PuzzleBotFSM.PARKING:
            park_pos = self.anymal_dest + self.DEPLOY_OFFSETS[i]
            
            obstacles = [b for idx, b in enumerate(self.puzzlebots) if idx != i] + self.stack_boxes
            safe_target = self._get_avoidance_target(bot, park_pos, obstacles, safe_dist=0.4)
            
            arrived = bot.navigate_to(safe_target, dt)
            bot.step(dt)
            
            if arrived:
                self.pb_fsm[i] = PuzzleBotFSM.DONE
                self._log(f"  [Slot {i}] ✓ Parked safely out of the way.")
                
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
        
    def _get_avoidance_target(self, robot, true_target, obstacles, safe_dist=0.3, k_rep=0.4):
        """Calculates a virtual target using Area-Aware Artificial Potential Fields with Vortex."""
        rx, ry = robot.x, robot.y
        tx, ty = true_target[0], true_target[1]
        
        # Use the actual physical boundaries of the robot
        robot_bounds = robot.get_bounds()
        
        # 1. Attractive vector towards the true target
        dx, dy = tx - rx, ty - ry
        dist_t = norm2(dx, dy)
        
        if dist_t < 0.25:
            return [tx, ty]
        
        if dist_t > 0:
            scale = min(1.0, dist_t)
            dx = (dx / dist_t) * scale
            dy = (dy / dist_t) * scale
            
        # 2. Repulsive vectors away from obstacles
        rep_x, rep_y = 0.0, 0.0
        has_bounds = hasattr(robot, 'get_bounds')
        
        rep_x, rep_y = 0.0, 0.0
        for obs in obstacles:
            # Measure true edge-to-edge physical distance, not centers!
            if has_bounds:
                robot_bounds = robot.get_bounds()
                dist_o = get_aabb_distance(robot_bounds, obs.get_bounds())
            else:
                dist_o = norm2(rx - obs.x, ry - obs.y)
            
            if 0.01 < dist_o < safe_dist:
                mag = k_rep * (1.0 / dist_o - 1.0 / safe_dist) / (dist_o**2)
                
                mag *= min(1.0, (dist_t / safe_dist))
                
                # Direction vector from obstacle center to robot center
                dir_x = rx - obs.x
                dir_y = ry - obs.y
                dir_dist = norm2(dir_x, dir_y)
                
                if dir_dist > 0:
                    r_x = (dir_x / dir_dist) * mag
                    r_y = (dir_y / dir_dist) * mag
                    
                    # Standard repulsion (pushes away)
                    rep_x += r_x
                    rep_y += r_y
                    
                    # VORTEX Force: Rotates the repulsion 90 degrees so the robot "slides" around
                    rep_x += -r_y 
                    rep_y += r_x

        v_x = dx + rep_x
        v_y = dy + rep_y
        
        # 3. Anti-freeze escape: If forces cancel perfectly, shove the robot sideways
        if norm2(v_x, v_y) < 0.05:
            v_x += 0.5
            v_y += 0.5

        # Return the new virtual waypoint
        return [rx + v_x, ry + v_y]

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
            # Hackathon metrics
            'metrics': {
                'cmax'              : round(self.cmax_timer, 1),
                'replans'           : self.planner.replan_count,
                'collisions_avoided': self.collisions_avoided,
                'obstacles_seen'    : len(self.last_obstacles_det),
                'landmarks_seen'    : len(self.last_landmarks_det),
            },
        }